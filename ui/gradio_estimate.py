import imageio
import numpy as np
import gradio as gr
from PIL import Image
import shutil
from main import PoseEstimation, Analysis
import os
estimator = PoseEstimation()
analyzer = Analysis()


def estimate(input_video):
    return estimator(input_video)


def analyze(mode):
    return analyzer(mode)


css = """
.gradio-container {
  background-color: rgba(242, 240, 225, 1);
}
.custom-textbox {
    background-color: rgba(244, 188, 65, 0.6);
    color: black;
}
"""

with gr.Blocks() as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center; 
            background-color: rgba(52, 102, 165, 1); border-radius: 10px;">
            <div style="flex: 1; padding: 20px;">
                <h3 style="font-size: 2em; margin-bottom: 0.5em; color: #ffffff">人体姿态估计在体育运动及教学中的实践应用</h3>
                <h5 style="margin: 0.3em; color: #ffffff">✨基于3D姿态估计技术的体育运动自动教学系统✨</h5>
            </div>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column():
            input_video = gr.Video(format="mp4", label="输入视频")
            mode = gr.Dropdown(label="动作模式", choices=["扣球🏐", "拦网🙌🏻"])
            with gr.Row():
                submit1 = gr.Button("姿态估计", elem_id="submit1")
                submit2 = gr.Button("动作分析", elem_id="submit2")

        with gr.Column():
            estimation = gr.Video(format="mp4", label="姿态估计结果", autoplay=True, scale=1)
            analysis = gr.Textbox(label="分析结果", elem_id="analysis", scale=1)


    def read_video(video):
        video_path = video  # 获取上传视频的临时路径
        save_path = "./demo/video/sample_video.mp4"  # 设置保存路径和文件名
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # 创建目录
        shutil.copy(video_path, save_path)  # 复制文件到指定路径
        print("upload successfully!!!!")
        #reader = imageio.get_reader(video)
        #fps = reader.get_meta_data()['fps']
        return video


    def read_image(image, size=512):
        return np.array(Image.fromarray(image).resize((size, size)))


    # 上传新视频时进行读取
    input_video.upload(
        read_video,
        input_video,
        input_video
    )
    # 点击姿态估计按钮
    submit1.click(
        estimate,
        input_video,
        estimation
    )
    # 点击动作分析按钮
    submit2.click(
        analyze,
        mode,
        analysis
    )

    # 示例
    gr.Markdown("## 示例")
    gr.Examples(
        examples="demo/example/input",
        inputs=input_video,
        outputs=estimation,
    )

demo.css = css
demo.launch(share=True)
