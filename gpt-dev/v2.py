import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 16
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embed = 64
n_head = 8
n_layer = 4
dropout = 0.2

print("Using device:", device)

torch.manual_seed(1337)  # for reproducibility

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string


def encode(s):
    """encode a string to a list of integers"""
    return [stoi[c] for c in s]


def decode(int_list):
    """decode a list of integers to a string"""
    return "".join([itos[i] for i in int_list])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]


def get_batch(split="train"):
    """generate a small batch of data of inputs x and targets y"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    """estimate the loss on the training and validation sets"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            # 修正：将数据移到正确的设备
            xb = xb.to(device)
            yb = yb.to(device)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

class Head(nn.Module):
    """A single attention head"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, H)
        q = self.query(x)  # (B, T, H)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, H) @ (B, H, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # apply causal mask
        wei = F.softmax(wei, dim=-1)  # normalize to probabilities
        v = self.value(x)  # (B, T, H)
        out = wei @ v  # (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, head_num, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_num)])
        self.proj = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        out = [h(x) for h in self.heads]  # list of (B, T, H)
        out = torch.cat(out, dim=-1)  # (B, T, n_heads * H)
        out = self.proj(out)  # (B, T, n_embed)
        return out

class FeedForward(nn.Module):
    """A simple feed-forward layer."""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),  # add dropout for regularization
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embed, head_num):
        super().__init__()
        head_size = n_embed // head_num
        self.self_attn = MultiHeadAttention(head_num, head_size)
        self.feed_fwd = FeedForward(n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))  # self-attention with residual connection
        x = x + self.feed_fwd(self.layer_norm2(x))  # feed-forward
        return x


class BigramLanguageModel(nn.Module):
    """A bigram language model"""

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, head_num=n_head) for _ in range(n_layer)])
        self.layer_norm_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
        
    def forward(self, idx, targets=None):
        batch_size, block_size = idx.shape

        tok_embed = self.token_embedding_table(idx)  # (batch_size, block_size, n_embed)
        pos_embed = self.positional_embedding(torch.arange(block_size, device=device))  # (block_size, n_embed)
        x = tok_embed + pos_embed  # (batch_size, block_size, n_embed)
        x = self.blocks(x)  # (batch_size, block_size, n_embed)
        logits = self.lm_head(x)  # (batch_size, block_size, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is the (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

        return idx


model = BigramLanguageModel().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # 修正：正确处理 get_batch 返回的元组
    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
