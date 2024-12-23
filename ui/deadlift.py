import numpy as np
import os
from benchpress_data import save_data, load_data


# 两脚介于与肩同宽和与髋同宽之间 肩宽/髋宽/脚宽
# 在不移动杠铃杆的前提下，弯曲膝盖，使小腿贴住杠铃杆。 小腿和杠铃距离
# 在不移动杠铃杆的前提下，挺胸，腰椎维持正常曲度，进入硬拉起始姿势。此外，还需保持头部中立位（既不抬头，也不低头），这样我们能更容易保持挺胸直背的姿势。 屁股肚脐脖子角度 额头脖子肚脐角度
# 肩胛骨、杠铃杆以及脚中心点在同一竖直平面内对齐，保持挺胸，腰椎维持正常曲度，肘关节伸直，双脚全脚掌着地 ？？
# 错误的硬拉起始姿势：a.杠铃杆位于脚中心点前方的位置：b.肩胛骨位于杠铃杆后方的位置：
# 第1个错误：驼背弯腰   第2个错误：手臂弯曲   第3个错误：杠铃脱离腿部   硬拉锁定姿势时腰椎过伸：

# 脚宽因子：脚宽**2/(肩宽*髋宽)  （开始时记录） feet_factor
# 小腿和杠铃距离  （开始/高点/结束时记录） leg_barbell_distance
# 屁股肚脐脖子标准向量点积  （开始/高点/结束时记录）butt_navel_neck_dot_product
# 额头脖子肚脐标准向量点积  （开始/高点/结束时记录）head_neck_navel_dot_product
# 手肘肩标准向量点积  （开始时记录） hand_elbow_shoulder_dot_product

# 选取关键点编号
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


def test_deadlift_pose(train=False):
    result = ""
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
        return "训练数据已保存"

    # 计算均值
    elbow_angle_mean = np.mean(elbow_angle_data)
    knee_angle_mean = np.mean(knee_angle_data)
    hand_eye_mean = np.mean(hand_eye_data)
    arm_angle_mean = np.mean(arm_angle_data)
    hand_track_mean = np.mean(hand_track)
    # 计算方差
    elbow_angle_var = np.var(elbow_angle_data)
    knee_angle_var = np.var(knee_angle_data)
    hand_eye_var = np.var(hand_eye_data)
    arm_angle_var = np.var(arm_angle_data)
    hand_track_var = np.var(hand_track)
    elbow_angle_std = np.sqrt(elbow_angle_var)
    knee_angle_std = np.sqrt(knee_angle_var)
    hand_eye_std = np.sqrt(hand_eye_var)
    arm_angle_std = np.sqrt(arm_angle_var)
    hand_track_std = np.sqrt(hand_track_var)

    result = ""

    if elbow_angle < elbow_angle_mean - 3 * elbow_angle_std or elbow_angle > elbow_angle_mean + 3 * elbow_angle_std:
        if elbow_angle < 90:
            result += "用户在卧推时，小臂与大臂呈锐角，可能导致肩关节的过度外旋，会增加肘关节的压力，且胸肌的激活会减少，三角肌和肩部的其他肌群可能会代偿发力。请尽量保持90度发力。"
        if elbow_angle > 90:
            result += "用户在卧推时，小臂与大臂呈钝角，肩关节可能会处于一个相对不稳定的状态，增加肩关节的压力，且胸肌的激活会减少，三角肌和肱三头肌可能会代偿发力。请尽量保持90度发力。"
    if knee_angle < knee_angle_mean - 3 * knee_angle_std:
        result += "用户在卧推时，膝盖弯曲角度过小，可能会导致下半身稳定性不足，膝盖角度过小会使得下半身的肌肉群（如股四头肌、臀肌等）无法有效地参与支撑，整个身体的稳定性降低，可能导致上半身发力不均衡，从而增加腰部和肩部的负担，进而增加受伤风险。同时由于卧推时需要控制杠铃的重力，腰椎处于一个较为紧张的状态，若下肢没有有效支撑，可能导致腰椎过度弯曲或腰部疼痛。"
    elif knee_angle > knee_angle_mean + 3 * knee_angle_std:
        result += "用户在卧推时，膝盖弯曲角度过大，臀部会被迫处于过度屈曲状态。这样会导致臀部肌肉参与过多的发力，从而可能导致臀部和下背部的过度紧张或疲劳,同时下肢的肌肉群（尤其是大腿和臀部）会过度用力，这可能会打乱下肢和上肢之间的协调性。在卧推时，正确的膝盖弯曲应当有助于传递力到上肢，而过度弯曲可能会干扰力量传递的流畅性，影响卧推的推力效率。"
    if hand_eye < hand_eye_mean - 3 * hand_eye_std or hand_eye > hand_eye_mean + 3 * hand_eye_std:
        result += "用户在我卧推时，当杠铃处于最高点时，视线没有在杠铃正下方,视线的方向通常会影响头部和脖部的姿势。如果视线偏离杠铃，可能导致头部过度伸展或下压，这会影响颈部和脊柱的稳定性。长时间的不正确头部姿势可能引起脖部和背部的不适或疼痛。"
    if arm_angle > arm_angle_mean + 3 * arm_angle_std:
        result += "用户在卧推时，双臂夹角过大会导致肩膀的外展角度过大，增加肩部关节的负担，尤其是肩关节前部的旋转袖肌群。这种姿势可能会导致肩部过度拉伸或受压，长期这样做容易引起肩部疼痛、炎症，甚至是肩袖撕裂等严重问题。如果双臂夹角过大，手肘过低，胸大肌的激活程度可能会减弱，因为胸肌在过度外展的情况下无法有效发挥作用。反而，三角肌和肩部的其他肌肉群会承担更多的压力，可能导致训练效果不理想，甚至造成肌肉不平衡。"
    if hand_track < hand_track_mean - 3 * hand_track_std or hand_track > hand_track_mean + 3 * hand_track_std:
        result += "用户在卧推时，手臂应当保持一条弧线上下升，避免直上直下，当你在卧推时，手肘会略微向外扩展，形成一个自然的弧线，这有助于保持肩关节的安全和稳定。直上直下的动作会让肩关节过度承受压力，增加受伤风险。如果手臂直上直下，肩膀的内旋角度会过大，肩部的前侧肌肉（如肩袖）会过度受力。手臂弯曲并沿着弧线升降时，可以让肩部承受的压力更加均匀，从而避免因过度压迫造成肩关节的不适或损伤。手臂沿着弧线升降有助于更好地激活胸大肌。在直上直下的动作中，胸肌的参与度较低，更多的压力会被转移到肩膀和三头肌上。而弧线运动能够有效地让胸大肌承受更多的负荷，增强训练效果。"

    if result == "":
        result += "用户卧推时挥臂动作标准，保持状态，继续训练，持续反馈！。"
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

‘
def test_head_neck_navel_dot_product(kpt):
    head = kpt[10]
    neck = kpt[8]
    navel = kpt[7]
    head_neck_vector = head - neck
    head_neck_vector = head_neck_vector / np.linalg.norm(head_neck_vector)
    navel_neck_vector = navel - neck
    navel_neck_vector = navel_neck_vector / np.linalg.norm(navel_neck_vector)
    return np.dot(head_neck_vector, navel_neck_vector)
