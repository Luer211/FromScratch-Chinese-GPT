import gradio as gr
import torch
import json
from model.gpt import GPT, GPTConfig
from utils.generate_core import encode, decode, generate

def load_model():
    with open("data/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    char2idx = vocab["char2idx"]
    idx2char = {int(k): v for k, v in vocab["idx2char"].items()}

    config = GPTConfig(
        vocab_size=len(char2idx),
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128
    )
    model = GPT(config)
    model.load_state_dict(torch.load("checkpoints/gpt_libei.pt", map_location="cpu"))
    model.eval()

    return model, char2idx, idx2char

def run_generate(prompt, temperature, top_k, max_tokens):
    model, char2idx, idx2char = load_model()

    input_ids = torch.tensor([encode(prompt, char2idx)], dtype=torch.long)
    output_ids = generate(model, input_ids, max_tokens, temperature, top_k)
    return decode(output_ids[0].tolist(), idx2char)

def generate_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## 输入一句开头，让模型继续写")

        with gr.Row():
            prompt = gr.Textbox(label="起始句", placeholder="如：白日依山尽")

        with gr.Row():
            temperature = gr.Slider(0.5, 1.5, value=1.0, label="Temperature-创造性")
            top_k = gr.Slider(1, 100, value=20, step=1, label="Top-k-采样范围")
            max_tokens = gr.Slider(10, 300, value=100, step=10, label="Max tokens-生成字符数")

        generate_btn = gr.Button("生成文本")
        result_box = gr.Textbox(label="生成结果")

        generate_btn.click(fn=run_generate, inputs=[prompt, temperature, top_k, max_tokens], outputs=result_box)

    return tab