import torch
import torch.nn as nn
import math

class GPTConfig:
    def __init__(self, vocab_size, block_size, n_layer=4, n_head=4, n_embd=128, dropout=0.1):
        self.vocab_size = vocab_size      
        self.block_size = block_size      
        self.n_layer = n_layer            
        self.n_head = n_head              
        self.n_embd = n_embd              
        self.dropout = dropout


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape 
        assert T <= self.config.block_size, "超出了 block_size 的限制"

        token_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding(pos)
        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits  

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0 

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)

        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "mask", torch.tril(torch.ones(config.block_size, config.block_size))
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size() 

        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) 

        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  

        out = out.transpose(1, 2).contiguous().view(B, T, C) 
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
