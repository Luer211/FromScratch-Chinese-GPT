import torch

def encode(s, char2idx):
    return [char2idx.get(c, 0) for c in s]

def decode(indices, idx2char):
    return ''.join([idx2char.get(i, '?') for i in indices])

def generate(model, idx, max_new_tokens=100, temperature=1.0, top_k=20):
    block_size = model.config.block_size
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
    return idx
