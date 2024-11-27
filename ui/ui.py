import gradio as gr


def analyze_action(action, video):
    # TODO: 添加封装好的函数
    if action == "深蹲":
        return "这是深蹲"
    elif action == "硬拉":
        return "这是硬拉"
    elif action == "卧推":
        return "这是卧推"
    assert False, "Invalid action"


with gr.Blocks() as demo:
    gr.Markdown("# 健身动作分析")
    with gr.Row():
        action = gr.Dropdown(choices=["深蹲", "硬拉", "卧推"], label="选择动作")
    video = gr.Video(label="上传视频")
    submit = gr.Button("分析")
    output = gr.Textbox(label="反馈建议")

    submit.click(analyze_action, inputs=[action, video], outputs=output)

demo.launch()
