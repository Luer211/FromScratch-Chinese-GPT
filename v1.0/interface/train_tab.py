import gradio as gr
import threading
from train.train_core import train_model
import os

log_text = ""

def start_training(n_layer, n_head, n_embd, batch_size, learning_rate, max_iters, dropout):
    ckpt_path = "checkpoints/gpt_libei.pt"
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
        print(f"已删除旧模型权重：{ckpt_path}")

    params = [n_layer, n_head, n_embd, batch_size, learning_rate, max_iters, dropout]
    
    global log_text
    log_text = ""

    def logger(msg):
        global log_text
        log_text += msg + "\n"

    threading.Thread(target=train_model, args=(params, logger)).start()
    return "训练已启动，旧模型已清除，日志持续刷新中..."

def get_log():
    return log_text

def train_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## 配置你的模型训练参数")

        with gr.Row():
            n_layer = gr.Slider(1, 12, value=4, step=1, label="Transformer layers - 层数")
            n_head = gr.Slider(1, 12, value=4, step=1, label="Attention heads - 注意力头数")
            n_embd = gr.Slider(64, 512, value=128, step=32, label="Embedding dim - 向量维度")

        with gr.Row():
            batch_size = gr.Slider(8, 256, value=64, step=8, label="Batch size - 批大小")
            learning_rate = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="Learning rate - 学习率")
            max_iters = gr.Slider(1000, 20000, value=5000, step=500, label="Max iterations - 训练轮数")

        dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Dropout - 丢弃率")

        start_button = gr.Button("启动训练")
        train_status = gr.Textbox(label="训练状态 / 错误输出")

        log_box = gr.Textbox(label="训练日志输出", lines=15, interactive=False)
        log_timer = gr.Button("刷新日志")

        inputs = [n_layer, n_head, n_embd, batch_size, learning_rate, max_iters, dropout]
        start_button.click(fn=start_training, inputs=inputs, outputs=train_status)
        log_timer.click(fn=get_log, outputs=log_box)

    return tab