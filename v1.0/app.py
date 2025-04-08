import gradio as gr
from interface.data_tab import data_tab
from interface.train_tab import train_tab
from interface.generate_tab import generate_tab

demo = gr.TabbedInterface(
    interface_list=[
        data_tab(),        
        train_tab(),       
        generate_tab()     
    ],
    tab_names=["数据准备", "模型训练", "文本生成"]
)

if __name__ == '__main__':
    demo.launch()