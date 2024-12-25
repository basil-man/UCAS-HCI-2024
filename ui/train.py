import sys
import os

sys.path.append(os.getcwd())
current_dir = os.getcwd()
# 构建要添加的目录路径
demo_dir = os.path.join(current_dir, 'demo')

# 将指定目录添加到 sys.path 中
sys.path.append(demo_dir)
from demo.vis import generate
from deadlift import test_deadlift_pose 
from benchpress import  test_benchpress_pose
from deepsquat import test_squat_pose 
import time
from vedio_pad import make_square

def test_block_pose():
    return "1"
OUTPUT_VIDEO_PATH = "./ui/cache/square_vedio.mp4"

class PoseEstimation():
    def __call__(self, video_path):
        processed_video_path = generate(video_path)
        return processed_video_path
    
def process_file(file_path,mode):
    # 对文件进行处理的逻辑
    print(f"Processing file: {file_path}")
    make_square(file_path, OUTPUT_VIDEO_PATH)

    output_path = pose_estimation(OUTPUT_VIDEO_PATH)
    directory = os.path.dirname(output_path)
    if mode == "硬拉":
        test_deadlift_pose(1,directory)
    elif mode == "deep_squat":
        test_squat_pose(directory)
    elif mode == "卧推":
        test_benchpress_pose(1,directory)

def process_directory(directory_path,mode):
    # 遍历目录中的每个文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path,mode)


if __name__ == "__main__":
     # 创建 PoseEstimation 类的实例
    pose_estimation = PoseEstimation()

    # 调用实例，就像调用函数一样   
    file_path = './ui/train/deadlift'
    mode = '硬拉'
    
    process_directory(file_path,mode)
    

    
    
    
    


