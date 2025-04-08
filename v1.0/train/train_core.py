import os
import json
import torch
import torch.nn as nn
from model.gpt import GPT, GPTConfig
from utils.data_utils import get_batch, load_bin_data

def train_model(params, logger=print):
    n_layer, n_head, n_embd, batch_size, learning_rate, max_iters, dropout = params
    block_size = 128  
    data_dir = "data"
    save_path = "checkpoints/gpt_libei.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    vocab_path = os.path.join(data_dir, "vocab.json")
    with open(vocab_path, encoding="utf-8") as f:
        vocab = json.load(f)
    char2idx = vocab["char2idx"]
    idx2char = {int(k): v for k, v in vocab["idx2char"].items()}
    vocab_size = len(char2idx)

    train_data = load_bin_data("train")
    val_data = load_bin_data("val")

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout
    )
    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    logger("模型初始化完成，开始训练...")

    for iter in range(int(max_iters)):
        model.train()
        x, y = get_batch(train_data, block_size, batch_size, device)

        logits = model(x)
        B, T, C = logits.shape
        loss = loss_fn(logits.view(B * T, C), y.view(B * T))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = get_batch(val_data, block_size, batch_size, device)
                val_logits = model(x_val)
                val_loss = loss_fn(val_logits.view(-1, C), y_val.view(-1))

            logger(f"[iter {iter}] train loss: {loss.item():.4f}, val loss: {val_loss.item():.4f}")
            torch.save(model.state_dict(), save_path)
            logger(f"模型已保存至 {save_path}")

    logger("训练完成！")
