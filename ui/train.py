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
    file_path = './ui/train/benchpress/benchpress4.mp4'
    output_path = pose_estimation(file_path)
    directory = os.path.dirname(output_path)
    print(test_benchpress_pose(1,directory))
    

    
    
    
    


