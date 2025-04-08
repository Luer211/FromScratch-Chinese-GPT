import gradio as gr
import os
from utils.tokenizer import Tokenizer

def run_tokenizer(file, block_size):
    if file is None:
        return "请上传文本文件"
    try:
        data_dir = "data"
        for f in ["train.bin", "val.bin", "meta.pkl", "uploaded.txt"]:
            file_path = os.path.join(data_dir, f)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"已删除旧数据文件：{file_path}")

        uploaded_path = os.path.join(data_dir, "uploaded.txt")
        import shutil
        shutil.copy(file.name, uploaded_path)

        tokenizer = Tokenizer(input_path=uploaded_path, save_dir=data_dir)
        tokenizer.process(block_size=block_size)

        return f"数据预处理完成！字符数：{tokenizer.vocab_size}"
    except Exception as e:
        return f"预处理失败：{str(e)}"

def data_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## 上传你的语料文件(纯文本txt)")

        with gr.Row():
            file_input = gr.File(label="上传 txt 文件", file_types=[".txt"])
            block_slider = gr.Slider(64, 512, value=128, step=16, label="block_size-上下文长度")

        run_button = gr.Button("开始预处理")
        output_info = gr.Textbox(label="预处理状态")

        run_button.click(fn=run_tokenizer, inputs=[file_input, block_slider], outputs=output_info)

    return tab
