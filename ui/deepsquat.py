import numpy as np
import os

def test_squat_pose():
    """
    评估深蹲动作的标准程度。
    :param output_dir: 包含2D和3D关键点数据的输出目录。
    :return: 动作评估反馈。
    """
    # 加载2D和3D关键点数据
    try:
        output_dir = './demo/output/sample_video/'
        keypoints_2d = np.load(os.path.join(output_dir, 'input_2D', 'keypoints.npz'), allow_pickle=True)['reconstruction']
        keypoints_3d = np.load(os.path.join(output_dir, 'output_3D', 'output_keypoints_3d.npz'), allow_pickle=True)['reconstruction']
    except FileNotFoundError as e:
        return f"文件加载失败: {e}"

    # 选取关键点编号
    HIP = 0   # 臀部
    LEFT_HIP = 1
    LEFT_KNEE = 2
    LEFT_FOOT = 3
    RIGHT_HIP = 4
    RIGHT_KNEE = 5
    RIGHT_FOOT = 6
    NAVEL = 7  # 肚脐
    NECK = 8   # 脖子
    HEAD = 10  # 额头

    # 深蹲的关键阶段: 下蹲到最低点
    squat_depth = np.argmin(keypoints_2d[0, :, HIP, 1])  # 找到臀部最低点的帧索引
    squat_frame_3d = keypoints_3d[squat_depth, :, :]

    # 提取3D关键点坐标
    hip_position = squat_frame_3d[HIP]
    left_knee_position = squat_frame_3d[LEFT_KNEE]
    right_knee_position = squat_frame_3d[RIGHT_KNEE]
    left_foot_position = squat_frame_3d[LEFT_FOOT]
    right_foot_position = squat_frame_3d[RIGHT_FOOT]
    neck_position = squat_frame_3d[NECK]
    head_position = squat_frame_3d[HEAD]

    # 评估指标
    result = ""

    # 1. 躯干前倾角度
    torso_vector = neck_position - hip_position
    torso_angle = np.arctan2(torso_vector[1], torso_vector[2]) * 180 / np.pi
    if torso_angle < 30 or torso_angle > 45:
        result += f"躯干前倾角度为{torso_angle:.2f}度，应在30-45度之间。\n"

    # 2. 膝盖不能超过脚尖
    left_knee_to_foot = left_knee_position[0] - left_foot_position[0]
    right_knee_to_foot = right_knee_position[0] - right_foot_position[0]
    if left_knee_to_foot > 0 or right_knee_to_foot > 0:
        result += "膝盖超过了脚尖，应保持膝盖在脚尖的垂直线上方或稍后位置。\n"

    # 3. 双脚间距和膝盖张开距离的比例
    foot_distance = np.linalg.norm(left_foot_position - right_foot_position)
    knee_distance = np.linalg.norm(left_knee_position - right_knee_position)
    ratio = knee_distance / foot_distance
    if ratio < 1.2 or ratio > 1.5:
        result += f"膝盖间距与脚间距的比例为{ratio:.2f}，应在1.2-1.5之间。\n"

    # 4. 肩膀与双脚的对齐关系
    shoulder_width = np.linalg.norm(squat_frame_3d[14] - squat_frame_3d[11])  # 左肩和右肩
    if abs(foot_distance - shoulder_width) > 0.1 * shoulder_width:
        result += "双脚间距应与肩膀宽度相当，请调整脚的摆放位置。\n"

    # 5. 臀部和脚后跟的距离
    hip_to_left_heel = np.linalg.norm(hip_position - left_foot_position)
    hip_to_right_heel = np.linalg.norm(hip_position - right_foot_position)
    avg_hip_to_heel = (hip_to_left_heel + hip_to_right_heel) / 2
    if avg_hip_to_heel < 0.3 or avg_hip_to_heel > 0.5:
        result += f"臀部到脚后跟的距离为{avg_hip_to_heel:.2f}米，应在0.3-0.5米之间。\n"

    # 6. 手臂位置检查（若涉及摆臂）
    # 可根据需求添加手臂检查逻辑

    # 如果没有问题，提供正面反馈
    if result == "":
        result = "深蹲动作标准，保持良好状态，继续训练！"

    return result
