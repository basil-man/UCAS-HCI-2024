import gradio as gr
from main import analyze_video


def analyze_action(action, video):
    # 调用封装好的函数 analyze_video
    result = analyze_video(video, action)

    print(f"分析结果: {result}")
    
    if action == "深蹲":
        return f"深蹲分析结果: {result}"
    elif action == "硬拉":
        return f"硬拉分析结果: {result}"
    elif action == "卧推":
        return f"卧推分析结果: {result}"
    else:
        return "未知动作"


with gr.Blocks() as demo:
    gr.Markdown("# 健身动作分析")
    with gr.Row():
        action = gr.Dropdown(choices=["深蹲", "硬拉", "卧推"], label="选择动作")
    video = gr.Video(label="上传视频")
    submit = gr.Button("分析")
    output = gr.Textbox(label="反馈建议")

    submit.click(fn=analyze_action, inputs=[action, video], outputs=output)

demo.launch()
