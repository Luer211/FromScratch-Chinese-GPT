import os
import json
import numpy as np
from pathlib import Path 

class Tokenizer:
    def __init__(self, input_path, save_dir, train_ratio=0.9):
        
        self.input_path = Path(input_path)
        self.save_dir = Path(save_dir)
        self.train_ratio = train_ratio

        self.text = self.input_path.read_text(encoding="utf-8")

        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print(f"共识别出字符数(vocab size):{self.vocab_size}")

        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

        self.save_vocab()

    def save_vocab(self):
        vocab = {
            "char2idx": self.char2idx,
            "idx2char": self.idx2char,
        }
        (self.save_dir / "vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2), 
            encoding="utf-8"
        )
        print("字符映射已保存:vocab.json")

    def encode(self,s):
        return [self.char2idx[c] for c in s]

    def decode(self,ids):
        return ''.join([self.idx2char[i] for i in ids])

    def process(self, block_size=128):
        self.block_size = block_size  

        data = np.array(self.encode(self.text), dtype=np.uint16)

        split_idx = int(len(data) * self.train_ratio)
        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_data.tofile(self.save_dir / "train.bin")
        val_data.tofile(self.save_dir / "val.bin")

        print(f"数据编码完成:train.bin ({len(train_data)}), val.bin ({len(val_data)})")

if __name__ == "__main__":
    tokenizer = Tokenizer(
        input_path="data/LiBai_cleaned.txt",  
        save_dir="data"                     
    )
    tokenizer.process()