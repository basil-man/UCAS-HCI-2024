from vis import generate
from lib.spike import test_spike_pose,test_block_pose
import time
class PoseEstimation():
    def __call__(self, input):
        generate()
        return './demo/output/sample_video/output_video.mp4'


class Analysis():
    def __call__(self, mode):
        if mode == "扣球🏐":
            return test_spike_pose()
        else:
            return test_block_pose()
