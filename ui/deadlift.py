import numpy as np
import os


def test_deadlift_pose():
    """
    评估硬拉动作的标准程度。
    :param output_dir: 包含2D和3D关键点数据的输出目录。
    :return: 动作评估反馈。
    """
    # 加载2D和3D关键点数据
    try:
        output_dir = "./demo/output/sample_video/"
        keypoints_2d = np.load(os.path.join(output_dir, "input_2D", "keypoints.npz"), allow_pickle=True)[
            "reconstruction"
        ]
        keypoints_3d = np.load(os.path.join(output_dir, "output_3D", "output_keypoints_3d.npz"), allow_pickle=True)[
            "reconstruction"
        ]
    except FileNotFoundError as e:
        return f"文件加载失败: {e}"
