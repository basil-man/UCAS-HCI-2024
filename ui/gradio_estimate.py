import gradio as gr
import os
import shutil
from main import PoseEstimation, Analysis
from vedio_pad import make_square
import threading
import cv2
import time

# 创建核心逻辑的实例
estimator = PoseEstimation()
analyzer = Analysis()

# 文件保存路径
UPLOAD_DIR = "./demo/video/"
OUTPUT_VIDEO_PATH = "./ui/cache/square_vedio.mp4"
OUTPUT_DIR = "./demo/output/sample_video/"


# 处理上传视频
def read_video(video_path):
    save_path = os.path.join(UPLOAD_DIR, "uploaded_video.mp4")
    square_path = os.path.join(UPLOAD_DIR, "uploaded_video_square.mp4")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    shutil.copy(video_path, save_path)  # 保存上传的视频到指定路径
    make_square(save_path, square_path)  # 转换为正方形视频
    return square_path  # 返回正方形视频路径


# 调用姿态估计逻辑
def estimate(video_path):
    # 调用姿态估计，生成含关节标注的视频
    make_square(video_path, OUTPUT_VIDEO_PATH)
    output_path = estimator(OUTPUT_VIDEO_PATH)
    if not os.path.exists(output_path):
        return "Error: 姿态估计失败，未生成视频。"

    # 读取结果文件
    result_file_path = os.path.join(OUTPUT_DIR, "result.txt")
    if not os.path.exists(result_file_path):
        return "Error: 结果文件未找到。"

    with open(result_file_path, "r") as f:
        result = f.read()

    return output_path, result  # 返回处理后的视频路径和结果


# 调用动作分析逻辑
def analyze(video_path, mode):
    # 分析动作模式，返回文字结果
    analysis_result = analyzer(video_path,mode)
    if not analysis_result:
        return "Error: 动作分析失败。"
    return analysis_result


# Gradio 界面设计
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
            input_video = gr.Video(format="mp4", label="上传视频")
            mode = gr.Dropdown(
                label="动作模式", 
                choices=["卧推", "硬拉","深蹲"]
            )
            with gr.Row():
                submit1 = gr.Button("姿态估计")
                submit2 = gr.Button("动作分析")

        with gr.Column():
            estimation = gr.Video(label="姿态估计结果", autoplay=True)
            analysis = gr.Textbox(label="分析结果")

    # 按钮点击逻辑
    submit1.click(fn=lambda video_path: estimate(video_path)[0], inputs=input_video, outputs=estimation)
    submit2.click(
        fn=lambda video_path, mode: [estimate(video_path)[0], analyze(video_path, mode)],
        inputs=[input_video, mode],
        outputs=[estimation, analysis],
    )


demo.launch(share=True)
