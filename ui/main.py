import argparse
import os
import sys
import time

# 添加当前目录和 demo 子目录到路径
sys.path.append(os.getcwd())
demo_dir = os.path.join(os.getcwd(), "demo")
sys.path.append(demo_dir)

from demo.vis import generate
from deadlift import test_deadlift_pose as deadlift_pose
from benchpress import test_benchpress_pose as benchpress_pose
from deepsquat import test_squat_pose as deepsquat_pose
from GPT_api import generate_prompt as gpt_prompt

# Function definitions


def test_block_pose():
    """Fallback analysis function."""
    return "1"


class PoseEstimation:
    """Handles pose estimation."""

    def __call__(self, video_path):
        processed_video_path = generate(video_path)
        return processed_video_path


class Analysis:
    """Performs analysis based on mode."""

    def __call__(self, mode):
        if mode == "硬拉":
            return deadlift_pose()
        elif mode == "deep_squat":
            return deepsquat_pose()
        elif mode == "卧推":
            return benchpress_pose(0)
        else:
            return test_block_pose()


def analyze_video(video_path, mode):
    """Runs pose estimation and analysis."""
    pose_estimation = PoseEstimation()
    pose_estimation(video_path)

    analysis = Analysis()
    return analysis(mode)


# Main script execution
if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Pose Estimation Analysis")
    parser.add_argument("mode", type=str, help="The mode of analysis (e.g., 硬拉, deep_squat, 卧推)")
    parser.add_argument("--video", type=str, default="./demo/video/sample_video.mp4", help="Path to the input video")
    parser.add_argument("--api", type=bool, default=False, help="Whether to use the GPT-4 API for analysis")
    args = parser.parse_args()

    # Start processing
    print("Starting pose estimation and analysis...")
    result = analyze_video(args.video, args.mode)
    if args.api:
        result += "\n[GPT 4 建议]：\n"
        result += gpt_prompt(result)
    # Output the result
    print("Analysis result:", result)
