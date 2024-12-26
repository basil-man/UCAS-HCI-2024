import numpy as np
import os
from benchpress_data import save_data, load_data


def test_benchpress_pose(train=False, video_path="./demo/output/square_vedio/"):
    result = ""
    output_dir = video_path
    keypoints_2d = np.load(output_dir + "/input_2D/keypoints.npz", allow_pickle=True)["reconstruction"]
    keypoints_3d = np.load(output_dir + "/output_3D/output_keypoints_3d.npz", allow_pickle=True)["reconstruction"]

    start_idx = -1
    highest_idx = -1
    finish_idx = -1
    right_elbow_angle_min = 180
    left_elbow_angle_min = 180
    right_elbow_angle_max = 0
    left_elbow_angle_max = 0
    for i in range(keypoints_3d.shape[0]):
        kpt = keypoints_3d[i, :, :]
        right_elbow_angle, left_elbow_angle = test_arm_straight(kpt)
        if right_elbow_angle + left_elbow_angle > right_elbow_angle_max + left_elbow_angle_max:
            highest_idx = i
            right_elbow_angle_max = right_elbow_angle
            left_elbow_angle_max = left_elbow_angle
        if right_elbow_angle + left_elbow_angle < right_elbow_angle_min + left_elbow_angle_min:
            finish_idx = i
            right_elbow_angle_min = right_elbow_angle
            left_elbow_angle_min = left_elbow_angle

    # print("highest_idx is ",highest_idx,"horizon_idx is ",horizon_idx)
    if highest_idx == -1 or finish_idx == -1:
        return "卧推动作不完整，视频录制过短，请重新录制视频，确保上传完整的卧推动作！"

    highesttime = keypoints_3d[highest_idx, :, :]
    finishtime = keypoints_3d[finish_idx, :, :]

    # 输入hkpt和lkpt是二维数组，第一维为时间帧，第二维为坐标 (x, y, z)
    # x表示横向，右手为负，y表示纵向高度，越高值越小，z表示前后，越靠前值越小
    # print(hkpt)
    # print(lkpt)
    # 目的是通过test_highest_info和test_arm_speed的返回值比对标准姿势和测试姿势的差别
    right_elbow_angle, left_elbow_angle = test_arm_straight(finishtime)  # head_highest_diffenence_height
    elbow_angle = (right_elbow_angle + left_elbow_angle) / 2
    knee_angle = test_leg_pose(highesttime)  # leftarm_highest_difference_height
    hand_eye = test_hand_eye(finishtime)
    arm_angle = test_arm_angle(finishtime)
    hand_track = test_hand_track(highest_idx, finish_idx, keypoints_3d)

    elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track_data = load_data()

    if train == 1:
        elbow_angle_data = np.append(elbow_angle_data, elbow_angle)
        knee_angle_data = np.append(knee_angle_data, knee_angle)
        hand_eye_data = np.append(hand_eye_data, hand_eye)
        arm_angle_data = np.append(arm_angle_data, arm_angle)
        hand_track_data = np.append(hand_track_data, hand_track)
        save_data(elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track)
        return "训练数据已保存"

    # 计算均值
    elbow_angle_mean = np.mean(elbow_angle_data)
    knee_angle_mean = np.mean(knee_angle_data)
    hand_eye_mean = np.mean(hand_eye_data)
    arm_angle_mean = np.mean(arm_angle_data)
    hand_track_mean = np.mean(hand_track_data)
    # 计算方差
    elbow_angle_var = np.var(elbow_angle_data)
    knee_angle_var = np.var(knee_angle_data)
    hand_eye_var = np.var(hand_eye_data)
    arm_angle_var = np.var(arm_angle_data)
    hand_track_var = np.var(hand_track_data)
    elbow_angle_std = np.sqrt(elbow_angle_var)
    knee_angle_std = np.sqrt(knee_angle_var)
    hand_eye_std = np.sqrt(hand_eye_var)
    arm_angle_std = np.sqrt(arm_angle_var)
    hand_track_std = np.sqrt(hand_track_var)

    result = ""

    if elbow_angle < elbow_angle_mean - 3 * elbow_angle_std or elbow_angle > elbow_angle_mean + 3 * elbow_angle_std:
        if elbow_angle < 90:
            result += "用户在卧推时，小臂与大臂呈锐角，可能导致肩关节的过度外旋，会增加肘关节的压力，且胸肌的激活会减少，三角肌和肩部的其他肌群可能会代偿发力。请尽量保持90度发力。\n"
        if elbow_angle > 90:
            result += "用户在卧推时，小臂与大臂呈钝角，肩关节可能会处于一个相对不稳定的状态，增加肩关节的压力，且胸肌的激活会减少，三角肌和肱三头肌可能会代偿发力。请尽量保持90度发力。\n"
    if knee_angle < knee_angle_mean - 3 * knee_angle_std:
        result += "用户在卧推时，膝盖弯曲角度过小，可能会导致下半身稳定性不足，膝盖角度过小会使得下半身的肌肉群（如股四头肌、臀肌等）无法有效地参与支撑，整个身体的稳定性降低，可能导致上半身发力不均衡，从而增加腰部和肩部的负担，进而增加受伤风险。同时由于卧推时需要控制杠铃的重力，腰椎处于一个较为紧张的状态，若下肢没有有效支撑，可能导致腰椎过度弯曲或腰部疼痛。\n"
    elif knee_angle > knee_angle_mean + 3 * knee_angle_std:
        result += "用户在卧推时，膝盖弯曲角度过大，臀部会被迫处于过度屈曲状态。这样会导致臀部肌肉参与过多的发力，从而可能导致臀部和下背部的过度紧张或疲劳,同时下肢的肌肉群（尤其是大腿和臀部）会过度用力，这可能会打乱下肢和上肢之间的协调性。在卧推时，正确的膝盖弯曲应当有助于传递力到上肢，而过度弯曲可能会干扰力量传递的流畅性，影响卧推的推力效率。\n"
    if hand_eye < hand_eye_mean - 3 * hand_eye_std or hand_eye > hand_eye_mean + 3 * hand_eye_std:
        result += "用户在我卧推时，当杠铃处于最高点时，视线没有在杠铃正下方,视线的方向通常会影响头部和脖部的姿势。如果视线偏离杠铃，可能导致头部过度伸展或下压，这会影响颈部和脊柱的稳定性。长时间的不正确头部姿势可能引起脖部和背部的不适或疼痛。\n"
    if arm_angle > arm_angle_mean + 3 * arm_angle_std:
        result += "用户在卧推时，双臂夹角过大会导致肩膀的外展角度过大，增加肩部关节的负担，尤其是肩关节前部的旋转袖肌群。这种姿势可能会导致肩部过度拉伸或受压，长期这样做容易引起肩部疼痛、炎症，甚至是肩袖撕裂等严重问题。如果双臂夹角过大，手肘过低，胸大肌的激活程度可能会减弱，因为胸肌在过度外展的情况下无法有效发挥作用。反而，三角肌和肩部的其他肌肉群会承担更多的压力，可能导致训练效果不理想，甚至造成肌肉不平衡。\n"
    if hand_track < hand_track_mean - 3 * hand_track_std or hand_track > hand_track_mean + 3 * hand_track_std:
        result += "用户在卧推时，手臂应当保持一条弧线上下升，避免直上直下，当你在卧推时，手肘会略微向外扩展，形成一个自然的弧线，这有助于保持肩关节的安全和稳定。直上直下的动作会让肩关节过度承受压力，增加受伤风险。如果手臂直上直下，肩膀的内旋角度会过大，肩部的前侧肌肉（如肩袖）会过度受力。手臂弯曲并沿着弧线升降时，可以让肩部承受的压力更加均匀，从而避免因过度压迫造成肩关节的不适或损伤。手臂沿着弧线升降有助于更好地激活胸大肌。在直上直下的动作中，胸肌的参与度较低，更多的压力会被转移到肩膀和三头肌上。而弧线运动能够有效地让胸大肌承受更多的负荷，增强训练效果。\n"

    if result == "":
        result += "用户卧推时挥臂动作标准，保持状态，继续训练，持续反馈！。\n"
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
    return right_elbow_angle, left_elbow_angle


def test_leg_pose(kpt):
    right_knee = kpt[5]
    left_knee = kpt[2]
    hip = kpt[0]
    # 计算右膝与臀部的向量
    right_leg_vector = right_knee - hip
    # 计算左膝与臀部的向量
    left_leg_vector = left_knee - hip

    # 计算夹角
    knee_angle = calculate_angle(right_leg_vector, left_leg_vector)

    print(f"Knee angle: {knee_angle:.2f} degrees")
    return knee_angle


def test_arm_angle(kpt):
    right_elbow = kpt[15]
    left_elbow = kpt[12]
    neck = kpt[8]
    # 计算右肘到颈部的向量
    right_elbow_vector = right_elbow - neck
    # 计算左肘到颈部的向量
    left_elbow_vector = left_elbow - neck

    # 计算夹角
    arm_angle = calculate_angle(right_elbow_vector, left_elbow_vector)

    print(f"Right arm Left arm angle: {arm_angle:.2f} degrees")
    return arm_angle


def test_hand_eye(kpt):
    right_hand = kpt[16]
    left_hand = kpt[13]
    forehead = kpt[10]
    right_shoulder = kpt[14]
    left_shoulder = kpt[11]
    neck = kpt[8]

    # 计算右肘到颈部的向量
    right_shoulder_vector = right_shoulder - neck
    # 计算左肘到颈部的向量
    left_shoulder_vector = left_shoulder - neck

    # 计算法向量
    normal_vector = np.cross(right_shoulder_vector, left_shoulder_vector)

    hand_middle = (right_hand + left_hand) / 2

    forehead_middle_vector = forehead - hand_middle

    # 计算向量的夹角
    angle = calculate_angle(normal_vector, forehead_middle_vector)
    magnitude = np.linalg.norm(forehead_middle_vector)
    distance = np.sin(angle) * magnitude
    neck_forehead = forehead - neck

    return abs(distance) / abs(np.linalg.norm(neck_forehead))


def test_hand_track(highest_idx, finish_idx, keypoints_3d):
    highesttime = keypoints_3d[highest_idx, :, :]
    finishtime = keypoints_3d[finish_idx, :, :]

    highest_right_hand = highesttime[16]
    highest_left_hand = highesttime[13]
    finish_right_hand = finishtime[16]
    finish_left_hand = finishtime[13]
    highest_middle_hand = (highest_right_hand + highest_left_hand) / 2
    finish_middle_hand = (finish_right_hand + finish_left_hand) / 2

    finish_right_shoulder = finishtime[14]
    finish_left_shoulder = finishtime[11]
    finish_neck = finishtime[8]
    finish_hip = finishtime[0]
    right_shoulder_vector = finish_right_shoulder - finish_neck
    left_shoulder_vector = finish_left_shoulder - finish_neck
    body_length = np.linalg.norm(finish_neck - finish_hip)
    normal_vector = np.cross(right_shoulder_vector, left_shoulder_vector)
    angle = calculate_angle(normal_vector, finish_middle_hand - highest_middle_hand)
    magnitude = np.linalg.norm(finish_middle_hand - highest_middle_hand)
    distance = np.sin(angle) * magnitude
    return abs(distance) / abs(body_length)
