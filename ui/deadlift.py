import numpy as np
import os
from deadlift_data import save_data, load_data


# 两脚介于与肩同宽和与髋同宽之间 肩宽/髋宽/脚宽
# 在不移动杠铃杆的前提下，弯曲膝盖，使小腿贴住杠铃杆。 小腿和杠铃距离
# 在不移动杠铃杆的前提下，挺胸，腰椎维持正常曲度，进入硬拉起始姿势。此外，还需保持头部中立位（既不抬头，也不低头），这样我们能更容易保持挺胸直背的姿势。 屁股肚脐脖子角度 额头脖子肚脐角度
# 肩胛骨、杠铃杆以及脚中心点在同一竖直平面内对齐，保持挺胸，腰椎维持正常曲度，肘关节伸直，双脚全脚掌着地
# 错误的硬拉起始姿势：a.杠铃杆位于脚中心点前方的位置：b.肩胛骨位于杠铃杆后方的位置：
# 第1个错误：驼背弯腰   第2个错误：手臂弯曲   第3个错误：杠铃脱离腿部   硬拉锁定姿势时腰椎过伸：

# 脚宽因子：脚宽**2/(肩宽*髋宽)  （开始时记录） feet_factor
# 小腿和杠铃距离  （开始/高点/结束时记录） leg_barbell_distance
# 屁股肚脐脖子标准向量点积  （开始/高点/结束时记录）butt_navel_neck_dot_product
# 额头脖子肚脐标准向量点积  （开始/高点/结束时记录）head_neck_navel_dot_product
# 手肘肩标准向量点积  （开始时记录） hand_elbow_shoulder_dot_product

HIP = 0  # 臀部
LEFT_HIP = 1
LEFT_KNEE = 2
LEFT_FOOT = 3
RIGHT_HIP = 4
RIGHT_KNEE = 5
RIGHT_FOOT = 6
NAVEL = 7  # 肚脐
NECK = 8  # 脖子
HEAD = 10  # 额头
LEFT_HAND = 13  # 左手腕
LEFT_SHOULDER = 14  # 左肩
LEFT_ELBOW = 15  # 左肘


def test_deadlift_pose(train=False,video_path='./demo/output/sample_video/'):
    result = ""
    try:
        output_dir = video_path
        keypoints_2d = np.load(os.path.join(output_dir, "input_2D", "keypoints.npz"), allow_pickle=True)[
            "reconstruction"
        ]
        keypoints_3d = np.load(os.path.join(output_dir, "output_3D", "output_keypoints_3d.npz"), allow_pickle=True)[
            "reconstruction"
        ]
    except FileNotFoundError as e:
        return f"文件加载失败: {e}"

    start_idx = -1
    highest_idx = -1
    finish_idx = -1

    head_height = keypoints_2d[0, :, HEAD, 1]  # 额头的y坐标
    highest_idx = np.argmin(head_height)  # 找出头关节点最高的帧作为分析的目标帧
    start_left_hand_height = keypoints_2d[0, :highest_idx, LEFT_HAND, 1]
    end_left_hand_height = keypoints_2d[0, highest_idx:, LEFT_HAND, 1]
    start_idx = np.argmax(start_left_hand_height)
    finish_idx = np.argmax(end_left_hand_height) + highest_idx
    if highest_idx == -1 or finish_idx == -1 or start_idx == -1:
        return "硬拉动作不完整，视频录制过短，请重新录制视频，确保上传完整的硬拉动作！"

    start_kpt = keypoints_3d[start_idx, :, :]
    highest_kpt = keypoints_3d[highest_idx, :, :]
    finish_kpt = keypoints_3d[finish_idx, :, :]

    # 脚宽因子：脚宽**2/(肩宽*髋宽)  （开始时记录） feet_factor
    # 小腿和杠铃距离  （开始/高点/结束时记录） leg_barbell_distance
    # 屁股肚脐脖子标准向量点积  （开始/高点/结束时记录）butt_navel_neck_dot_product
    # 额头脖子肚脐标准向量点积  （开始/高点/结束时记录）head_neck_navel_dot_product
    # 胳膊夹角  （开始时记录） arm_angle

    feet_factor = test_feet_factor(start_kpt)
    start_leg_barbell_distance = test_leg_barbell_distance(start_kpt)
    heighest_leg_barbell_distance = test_leg_barbell_distance(highest_kpt)
    finish_leg_barbell_distance = test_leg_barbell_distance(finish_kpt)
    start_butt_navel_neck_dot_product = test_butt_navel_neck_dot_product(start_kpt)
    heighest_butt_navel_neck_dot_product = test_butt_navel_neck_dot_product(highest_kpt)
    finish_butt_navel_neck_dot_product = test_butt_navel_neck_dot_product(finish_kpt)
    start_head_neck_navel_dot_product = test_head_neck_navel_dot_product(start_kpt)
    heighest_head_neck_navel_dot_product = test_head_neck_navel_dot_product(highest_kpt)
    finish_head_neck_navel_dot_product = test_head_neck_navel_dot_product(finish_kpt)
    arm_angle = test_arm_straight(start_kpt)

    (
        feet_factor_data,
        start_leg_barbell_distance_data,
        heighest_leg_barbell_distance_data,
        finish_leg_barbell_distance_data,
        start_butt_navel_neck_dot_product_data,
        heighest_butt_navel_neck_dot_product_data,
        finish_butt_navel_neck_dot_product_data,
        start_head_neck_navel_dot_product_data,
        heighest_head_neck_navel_dot_product_data,
        finish_head_neck_navel_dot_product_data,
        arm_angle,
    ) = load_data()

    if train == True:
        feet_factor_data = np.append(feet_factor_data, feet_factor)
        start_leg_barbell_distance_data = np.append(start_leg_barbell_distance_data, start_leg_barbell_distance)
        heighest_leg_barbell_distance_data = np.append(
            heighest_leg_barbell_distance_data, heighest_leg_barbell_distance
        )
        finish_leg_barbell_distance_data = np.append(finish_leg_barbell_distance_data, finish_leg_barbell_distance)
        start_butt_navel_neck_dot_product_data = np.append(
            start_butt_navel_neck_dot_product_data, start_butt_navel_neck_dot_product
        )
        heighest_butt_navel_neck_dot_product_data = np.append(
            heighest_butt_navel_neck_dot_product_data, heighest_butt_navel_neck_dot_product
        )
        finish_butt_navel_neck_dot_product_data = np.append(
            finish_butt_navel_neck_dot_product_data, finish_butt_navel_neck_dot_product
        )
        start_head_neck_navel_dot_product_data = np.append(
            start_head_neck_navel_dot_product_data, start_head_neck_navel_dot_product
        )
        heighest_head_neck_navel_dot_product_data = np.append(
            heighest_head_neck_navel_dot_product_data, heighest_head_neck_navel_dot_product
        )
        finish_head_neck_navel_dot_product_data = np.append(
            finish_head_neck_navel_dot_product_data, finish_head_neck_navel_dot_product
        )
        arm_angle_data = np.append(arm_angle_data, arm_angle)
        save_data(
            feet_factor_data,
            start_leg_barbell_distance_data,
            heighest_leg_barbell_distance_data,
            finish_leg_barbell_distance_data,
            start_butt_navel_neck_dot_product_data,
            heighest_butt_navel_neck_dot_product_data,
            finish_butt_navel_neck_dot_product_data,
            start_head_neck_navel_dot_product_data,
            heighest_head_neck_navel_dot_product_data,
            finish_head_neck_navel_dot_product_data,
            arm_angle_data,
        )
        return "训练数据已保存"

    # 计算均值
    feet_factor_mean = np.mean(feet_factor_data)
    start_leg_barbell_distance_mean = np.mean(start_leg_barbell_distance_data)
    heighest_leg_barbell_distance_mean = np.mean(heighest_leg_barbell_distance_data)
    finish_leg_barbell_distance_mean = np.mean(finish_leg_barbell_distance_data)
    start_butt_navel_neck_dot_product_mean = np.mean(start_butt_navel_neck_dot_product_data)
    heighest_butt_navel_neck_dot_product_mean = np.mean(heighest_butt_navel_neck_dot_product_data)
    finish_butt_navel_neck_dot_product_mean = np.mean(finish_butt_navel_neck_dot_product_data)
    start_head_neck_navel_dot_product_mean = np.mean(start_head_neck_navel_dot_product_data)
    heighest_head_neck_navel_dot_product_mean = np.mean(heighest_head_neck_navel_dot_product_data)
    finish_head_neck_navel_dot_product_mean = np.mean(finish_head_neck_navel_dot_product_data)
    arm_angle_mean = np.mean(arm_angle_data)
    # 计算标准差
    feet_factor_std = np.std(feet_factor_data)
    start_leg_barbell_distance_std = np.std(start_leg_barbell_distance_data)
    heighest_leg_barbell_distance_std = np.std(heighest_leg_barbell_distance_data)
    finish_leg_barbell_distance_std = np.std(finish_leg_barbell_distance_data)
    start_butt_navel_neck_dot_product_std = np.std(start_butt_navel_neck_dot_product_data)
    heighest_butt_navel_neck_dot_product_std = np.std(heighest_butt_navel_neck_dot_product_data)
    finish_butt_navel_neck_dot_product_std = np.std(finish_butt_navel_neck_dot_product_data)
    start_head_neck_navel_dot_product_std = np.std(start_head_neck_navel_dot_product_data)
    heighest_head_neck_navel_dot_product_std = np.std(heighest_head_neck_navel_dot_product_data)
    finish_head_neck_navel_dot_product_std = np.std(finish_head_neck_navel_dot_product_data)
    arm_angle_std = np.std(arm_angle_data)

    # 两脚介于与肩同宽和与髋同宽之间 肩宽/髋宽/脚宽
    # 在不移动杠铃杆的前提下，弯曲膝盖，使小腿贴住杠铃杆。 小腿和杠铃距离
    # 在不移动杠铃杆的前提下，挺胸，腰椎维持正常曲度，进入硬拉起始姿势。此外，还需保持头部中立位（既不抬头，也不低头），这样我们能更容易保持挺胸直背的姿势。 屁股肚脐脖子角度 额头脖子肚脐角度
    # 肩胛骨、杠铃杆以及脚中心点在同一竖直平面内对齐，保持挺胸，腰椎维持正常曲度，肘关节伸直，双脚全脚掌着地 ？？
    # 错误的硬拉起始姿势：a.杠铃杆位于脚中心点前方的位置：b.肩胛骨位于杠铃杆后方的位置：
    # 第1个错误：驼背弯腰   第2个错误：手臂弯曲   第3个错误：杠铃脱离腿部   硬拉锁定姿势时腰椎过伸：

    result = ""

    if feet_factor < feet_factor_mean - 3 * feet_factor_std:
        result += "用户在硬拉时，两脚间距过小，可能导致上半身稳定性不足，使得上半身的肌肉群（如肩部、背部、核心肌群等）无法有效地参与支撑，整个身体的稳定性降低，进而增加受伤风险。建议两脚介于与肩同宽和与髋同宽之间。\n"
    elif feet_factor > feet_factor_mean + 3 * feet_factor_std:
        result += "用户在硬拉时，两脚间距过大，会增加膝关节和髋关节的负担，提升受伤风险。并且加大下背部的压力，容易导致腰部受伤。两脚介于与肩同宽和与髋同宽之间。\n"
    if (
        start_leg_barbell_distance > start_leg_barbell_distance_mean + 3 * start_leg_barbell_distance_std
        or heighest_leg_barbell_distance > heighest_leg_barbell_distance_mean + 3 * heighest_leg_barbell_distance_std
        or finish_leg_barbell_distance > finish_leg_barbell_distance_mean + 3 * finish_leg_barbell_distance_std
    ):
        result += "用户在硬拉起始姿势时，小腿和杠铃的距离过大，可能导致腰部过度前倾，增加腰部的压力，容易导致腰部受伤。同时，在后续髋部抬高的过程中，杠铃杆不再紧贴胫骨，处于失衡状态，增大对腰部稳定性的要求，从而增加腰部受伤风险。建议硬拉全程，杠铃杆始终贴着腿部上下移动。\n"
    if (
        start_butt_navel_neck_dot_product
        > start_butt_navel_neck_dot_product_mean + 3 * start_butt_navel_neck_dot_product_std
        or start_butt_navel_neck_dot_product
        < start_butt_navel_neck_dot_product_mean - 3 * start_butt_navel_neck_dot_product_std
    ):
        result += "用户在硬拉起始动作时，上半身并未保持挺胸收腹，而是弯腰驼背，这会使得腰部承受更大的压力，增加腰关节受伤风险。建议在硬拉起始动作时，上半身保持挺胸收腹，腰部保持正常曲度。\n"
    if (
        heighest_butt_navel_neck_dot_product
        < heighest_butt_navel_neck_dot_product_mean - 3 * heighest_butt_navel_neck_dot_product_std
        or heighest_butt_navel_neck_dot_product
        > heighest_butt_navel_neck_dot_product_mean + 3 * heighest_butt_navel_neck_dot_product_std
    ):
        result += "用户在硬拉高点时，上半身并未保持挺胸收腹，而是弯腰驼背，这会给肩部关节和软组织带来更大的压力，增加受伤风险。建议在硬拉高点时，上半身保持挺胸收腹，腰部保持正常曲度。\n"
    if (
        finish_butt_navel_neck_dot_product
        < finish_butt_navel_neck_dot_product_mean - 3 * finish_butt_navel_neck_dot_product_std
        or finish_butt_navel_neck_dot_product
        > finish_butt_navel_neck_dot_product_mean + 3 * finish_butt_navel_neck_dot_product_std
    ):
        result += "用户在硬拉完成动作时，上半身并未保持挺胸收腹，而是弯腰驼背，这表明杠铃下放过程中，可能存在弯腰的情况，对腰关节造成更大的压力，存在受伤风险\n"
    if arm_angle > arm_angle_mean + 3 * arm_angle_std or arm_angle < arm_angle_mean - 3 * arm_angle_std:
        result += "用户在硬拉时，手臂未保持伸直，而是弯曲，这会增加肩部和肘部的压力，增加受伤风险。建议在硬拉时，手臂保持伸直，不要弯曲。\n"
    if (
        start_head_neck_navel_dot_product
        < start_head_neck_navel_dot_product_mean - 3 * start_head_neck_navel_dot_product_std
        or start_head_neck_navel_dot_product
        > start_head_neck_navel_dot_product_mean + 3 * start_head_neck_navel_dot_product_std
        or heighest_head_neck_navel_dot_product
        < heighest_head_neck_navel_dot_product_mean - 3 * heighest_head_neck_navel_dot_product_std
        or heighest_head_neck_navel_dot_product
        > heighest_head_neck_navel_dot_product_mean + 3 * heighest_head_neck_navel_dot_product_std
        or finish_head_neck_navel_dot_product
        < finish_head_neck_navel_dot_product_mean - 3 * finish_head_neck_navel_dot_product_std
        or finish_head_neck_navel_dot_product
        > finish_head_neck_navel_dot_product_mean + 3 * finish_head_neck_navel_dot_product_std
    ):
        result += "用户在硬拉时，应保持抬头，不要低头，这样可以更好地保持挺胸直背的姿势，减少腰部受力，降低受伤风险。\n"
    if result == "":
        result += "用户硬拉时动作标准，保持状态，继续训练，持续反馈！。"
    return result


def calculate_angle(v1, v2):
    # 计算两个向量的点积
    dot_product = np.dot(v1, v2)
    # 计算两个向量的模
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # 检查向量模是否为零
    if norm_v1 == 0 or norm_v2 == 0:
        raise ValueError("输入的向量不能为零向量")

    # 计算夹角的余弦值，并确保值在[-1, 1]范围内
    cos_angle = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
    # 计算夹角（弧度）
    angle = np.arccos(cos_angle)
    # 将弧度转换为角度
    angle_degrees = np.degrees(angle)
    return angle_degrees


def test_arm_straight(kpt):
    right_arm = kpt[14]
    right_elbow = kpt[15]
    right_wrist = kpt[16]
    left_arm = kpt[11]
    left_elbow = kpt[12]
    left_wrist = kpt[13]
    # 计算右肩肘与手肘的向量
    right_upper_arm_vector = right_arm - right_elbow
    right_forearm_vector = right_wrist - right_elbow

    # 计算左肩肘与手肘的向量
    left_upper_arm_vector = left_arm - left_elbow
    left_forearm_vector = left_wrist - left_elbow

    # 计算右肩肘与手肘的夹角
    right_elbow_angle = calculate_angle(right_upper_arm_vector, right_forearm_vector)

    # 计算左肩肘与手肘的夹角
    left_elbow_angle = calculate_angle(left_upper_arm_vector, left_forearm_vector)

    print(f"Right elbow angle: {right_elbow_angle:.2f} degrees")
    print(f"Left elbow angle: {left_elbow_angle:.2f} degrees")
    return (right_elbow_angle + left_elbow_angle) / 2


def test_feet_factor(kpt):
    lfeet = kpt[6]
    rfeet = kpt[3]
    rshoulder = kpt[14]
    lshoulder = kpt[11]
    lhip = kpt[4]
    rhip = kpt[1]
    feet_len = np.linalg.norm(lfeet - rfeet)
    shoulder_len = np.linalg.norm(lshoulder - rshoulder)
    hip_len = np.linalg.norm(lhip - rhip)
    return feet_len**2 / (shoulder_len * hip_len)


def test_leg_barbell_distance(kpt):
    def distance_point_to_line(point, line_start, line_end):
        line_vec = line_end - line_start
        point_vec = point - line_start
        line_len = np.linalg.norm(line_vec)
        line_unitvec = line_vec / line_len
        projection_length = np.dot(point_vec, line_unitvec)
        projection = line_start + projection_length * line_unitvec
        distance = np.linalg.norm(point - projection)
        return distance

    lfeet = kpt[6]
    rfeet = kpt[3]
    lhand = kpt[13]
    rhand = kpt[16]
    lknee = kpt[3]
    rknee = kpt[5]

    distance_lhand_to_lower_leg = distance_point_to_line(lhand, lfeet, lknee)
    distance_rhand_to_lower_leg = distance_point_to_line(rhand, rfeet, rknee)
    return (distance_lhand_to_lower_leg + distance_rhand_to_lower_leg) / 2


def test_butt_navel_neck_dot_product(kpt):
    butt = kpt[0]
    navel = kpt[7]
    neck = kpt[8]
    butt_navel_vector = butt - navel
    butt_navel_vector = butt_navel_vector / np.linalg.norm(butt_navel_vector)
    neck_navel_vector = neck - navel
    neck_navel_vector = neck_navel_vector / np.linalg.norm(neck_navel_vector)
    return np.dot(butt_navel_vector, neck_navel_vector)


def test_head_neck_navel_dot_product(kpt):
    head = kpt[10]
    neck = kpt[8]
    navel = kpt[7]
    head_neck_vector = head - neck
    head_neck_vector = head_neck_vector / np.linalg.norm(head_neck_vector)
    navel_neck_vector = navel - neck
    navel_neck_vector = navel_neck_vector / np.linalg.norm(navel_neck_vector)
    return np.dot(head_neck_vector, navel_neck_vector)
