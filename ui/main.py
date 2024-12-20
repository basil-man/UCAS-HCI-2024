import argparse
import sys
import os

sys.path.append(os.getcwd())
current_dir = os.getcwd()
# 构建要添加的目录路径
demo_dir = os.path.join(current_dir, 'demo')

# 将指定目录添加到 sys.path 中
sys.path.append(demo_dir)
from demo.vis import generate
from deadlift import test_spike_pose as deadlift_pose
from benchpress import test_spike_pose as benchpress_pose
from deepsquat import test_spike_pose as deepsquat_pose
import time

def test_block_pose():
    return "1"


class PoseEstimation():
    def __call__(self, input):
        generate(input)
        return './demo/output/sample_video/output_video.mp4'


class Analysis():
    def __call__(self, mode):
        if mode == "硬拉":
            return deadlift_pose()
        elif mode == "深蹲":
            return deepsquat_pose()
        elif mode == "卧推":
            return benchpress_pose()
        else:
            return test_block_pose()
        
def analyze_video(video_path,mode):
    # 创建 PoseEstimation 类的实例
    pose_estimation = PoseEstimation()

    # 调用实例，就像调用函数一样
    input_data = video_path
    pose_estimation(input_data)

    analysis = Analysis()
    
    return analysis(mode)

if __name__ == "__main__":
     # 创建 PoseEstimation 类的实例
    pose_estimation = PoseEstimation()

    # 调用实例，就像调用函数一样
    input_data = './demo/video/sample_video.mp4'
    

    analysis = Analysis()
    parser = argparse.ArgumentParser(description="Pose Estimation Analysis")
    parser.add_argument("mode", type=str, help="The mode of analysis (e.g., 硬拉, 深蹲, 卧推)")
    args = parser.parse_args()
    
    print(analysis(args.mode))
    

