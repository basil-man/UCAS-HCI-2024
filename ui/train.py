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


if __name__ == "__main__":
     # 创建 PoseEstimation 类的实例
    pose_estimation = PoseEstimation()

    # 调用实例，就像调用函数一样
    output_dir = './train/benchpress'

    for root, dirs, files in os.walk(output_dir):
        for file in files:
            file_path = os.path.join(root, file)
            pose_estimation(file_path)
            print(benchpress_pose(1))
    

    
    


