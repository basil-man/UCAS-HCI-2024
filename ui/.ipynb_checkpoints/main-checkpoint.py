import sys
import os

sys.path.append(os.getcwd())
current_dir = os.getcwd()
# 构建要添加的目录路径
demo_dir = os.path.join(current_dir, 'demo')

# 将指定目录添加到 sys.path 中
sys.path.append(demo_dir)
from demo.vis import generate
from deadlift import test_spike_pose,test_block_pose as deadlift_pose
import time
class PoseEstimation():
    def __call__(self, input):
        generate()
        return './demo/output/sample_video/output_video.mp4'


class Analysis():
    def __call__(self, mode):
        if mode == "扣球🏐":
            return test_spike_pose()
        else if
        else:
            return test_block_pose()

if __name__ == "__main__":
    # 创建 PoseEstimation 类的实例
    pose_estimation = PoseEstimation()

    # 调用实例，就像调用函数一样
    input_data = "some_input_data"
    output_path = pose_estimation(input_data)

    analysis = Analysis()

    
    print(analysis(mode))
    

