import gradio as gr
import os
import shutil
from main import PoseEstimation, Analysis
from vedio_pad import make_square
import atexit
from GPT_api import generate_prompt as generate_advice
import cv2
import threading
import time
import queue

# 创建核心逻辑的实例
estimator = PoseEstimation()
analyzer = Analysis()

# 文件保存路径
UPLOAD_DIR = "./demo/video/"
OUTPUT_VIDEO_PATH = "./ui/cache/square_vedio.mp4"
OUTPUT_DIR = "./demo/output/sample_video/"
REALTIME_VIDEO_PATH = os.path.join(UPLOAD_DIR, "realtime_video.mp4")

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
    remove_folder()
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


def estimate_and_analyze(video_path, mode, gpt_analysis):
    if not mode:
        raise gr.Error("请选择动作模式后再进行动作分析。")
    estimate(video_path)
    analyze_result = analyze(video_path, mode)
    if gpt_analysis:
        gpt_result = generate_advice(mode, analyze_result)
        return "./demo/output/square_vedio/square_vedio.mp4", analyze_result + "\n" + gpt_result
    else:
        return "./demo/output/square_vedio/square_vedio.mp4", analyze_result


def remove_folder():
    folder = "./demo/output/square_vedio/"
    if os.path.exists(folder):
        shutil.rmtree(folder)


atexit.register(remove_folder)

# 实时视频录制功能
recording = False
cap = None
out = None

def start_recording(mode):
    global recording, cap, out
    recording = True
    cap = cv2.VideoCapture(0)  
    if not cap.isOpened():
        print("Error: Could not open video capture.")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(REALTIME_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (frame_width, frame_height))
    threading.Thread(target=save_video_segments).start(mode)
    threading.Thread(target=record_video).start()

def stop_recording():
    global recording, cap, out
    recording = False
    if cap:
        cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

def save_video_segments(mode):
    global segment_index,recording
    while recording:
        time.sleep(10)  # 每隔10秒执行一次
        cap = cv2.VideoCapture(REALTIME_VIDEO_PATH)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        start_time = max(0, duration - 10)  # 计算开始时间，确保不超过视频长度
        start_frame = int(start_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        segment_path = f'realtime_video_segment_{segment_index}.mp4'
        out = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
        segment_index += 1
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        # 处理保存的视频片段
        return estimate_and_analyze(segment_path, mode, True)
        

def record_video():
    global recording, cap, out
    while recording:
        ret, frame = cap.read()
        print(ret)
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    stop_recording()

def periodic_output(output_queue):
    while True:
        time.sleep(10)  # 每隔10秒输出一次
        output_queue.put("定期输出的结果")

def update_ui(output_queue):
    results = []
    while not output_queue.empty():
        results.append(output_queue.get())
    return results

# Gradio 界面设计
with gr.Blocks(title="健身助手") as demo:
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
            mode = gr.Dropdown(label="动作模式", choices=["卧推", "硬拉", "深蹲"])
            gpt_analysis = gr.Checkbox(label="GPT分析", value=False)
            with gr.Row():
                submit1 = gr.Button("姿态估计")
                submit2 = gr.Button("动作分析")
            with gr.Row():
                start_button = gr.Button("开始录制")
                stop_button = gr.Button("停止录制")

        with gr.Column():
            estimation = gr.Video(label="姿态估计结果", autoplay=True)
            analysis = gr.Textbox(label="分析结果")
            periodic_output_component = gr.Textbox(label="实时分析结果")

    # 按钮点击逻辑
    submit1.click(fn=lambda video_path: estimate(video_path)[0], inputs=input_video, outputs=estimation)
    submit2.click(
        fn=estimate_and_analyze,
        inputs=[input_video, mode, gpt_analysis],
        outputs=[estimation, analysis],
    )

    start_button.click(fn=start_recording, inputs=mode, outputs=None)
    stop_button.click(fn=stop_recording, inputs=None, outputs=None)

    demo.load(fn=update_ui, inputs=None, outputs=periodic_output_component, every=10)

demo.launch(share=True, favicon_path="./ui/fig/logo.png")
