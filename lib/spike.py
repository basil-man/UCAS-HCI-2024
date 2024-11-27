import numpy as np
import os
def test_spike_pose():
    result = ""
    output_dir = './demo/output/sample_video/'
    keypoints_2d = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    keypoints_3d = np.load(output_dir + 'output_3D/output_keypoints_3d.npz', allow_pickle=True)['reconstruction']

    head_height = keypoints_2d[0,: , 10, 1] #额头的y坐标
    highest_idx = np.argmin(head_height) #找出头关节点最高的帧作为分析的目标帧
    larm_height = keypoints_2d[0,: , 13, 1] #额头的y坐标
    larm_highest_idx = np.argmin(larm_height)
    horizon_idx = highest_idx
    for i in range(keypoints_2d.shape[1]):
        if (i > highest_idx and keypoints_2d[0, i - 1, 16, 1] < keypoints_2d[0, i - 1, 14, 1] and keypoints_2d[0, i, 16, 1] > keypoints_2d[0, i, 14, 1]):
            horizon_idx = i
            break
    #print("highest_idx is ",highest_idx,"horizon_idx is ",horizon_idx)
    if(highest_idx == horizon_idx):
        return "扣球动作不完整，视频录制过短，请重新录制视频，确保上传完整的扣球动作！"

    head_highest = keypoints_3d[highest_idx, :, :]
    rarm_horizon = keypoints_3d[horizon_idx, :, :]
    larm_highest = keypoints_3d[larm_highest_idx, :, :]
    #输入hkpt和lkpt是二维数组，第一维为时间帧，第二维为坐标 (x, y, z)
    #x表示横向，右手为负，y表示纵向高度，越高值越小，z表示前后，越靠前值越小
    #print(hkpt)
    #print(lkpt)
    #目的是通过test_highest_info和test_arm_speed的返回值比对标准姿势和测试姿势的差别
    hh_diff,hh_angle = test_height_info(head_highest) #head_highest_diffenence_height
    lh_diff,lh_angle = test_height_info(larm_highest) #leftarm_highest_difference_height

    #print("athlete5 standard pose info: \nlh_diff:", lh_diff, "\tlh_angle:", lh_angle)
    #print("hh_diff:", hh_diff, "\thh_angle:", hh_angle)

    arm_speed = horizon_idx - highest_idx
    arm_angle, elbow_angle = test_arm_angle(rarm_horizon)
    #print("arm_angle:", arm_angle, "\telbow_angle:", elbow_angle)
    lh_diff_data = np.array([0.43961883, 0.43432528, 0.46093467, 0.3813015, 0.5607993])
    lh_angle_data = np.array(
        [15.486222981654413, 13.963960893721094, 17.062048958111383, 20.00910511506328, 9.218777294398617])
    hh_diff_data = np.array([0.50280684, 0.40918744, 0.4581399, 0.41242406, 0.6121681])
    hh_angle_data = np.array(
        [12.169783346599294, 12.971947464571311, 12.553325166719885, 18.25228174757277, 8.942303846331004])
    arm_angle_data = np.array(
        [11.69741786895216, 2.5779226241489153, 4.747299987571686, 24.00741516700715, 19.511018257477506])
    elbow_angle_data = np.array([87.74351, 75.067867, 68.095764, 69.01098, 61.46978])
    # 计算均值
    lh_diff_mean = np.mean(lh_diff_data)
    # 计算方差
    lh_diff_var = np.var(lh_diff_data)

    lh_angle_mean = np.mean(lh_angle_data)
    lh_angle_var = np.var(lh_angle_data)
    hh_diff_mean = np.mean(hh_diff_data)
    hh_diff_var = np.var(hh_diff_data)
    hh_angle_mean = np.mean(hh_angle_data)
    hh_angle_var = np.var(hh_angle_data)
    arm_angle_mean = np.mean(arm_angle_data)
    arm_angle_var = np.var(arm_angle_data)
    elbow_angle_mean = np.mean(elbow_angle_data)
    elbow_angle_var = np.var(elbow_angle_data)
    lh_diff_std = np.sqrt(lh_diff_var)
    lh_angle_std = np.sqrt(lh_angle_var)
    hh_diff_std = np.sqrt(hh_diff_var)
    hh_angle_std = np.sqrt(hh_angle_var)
    arm_angle_std = np.sqrt(arm_angle_var)
    elbow_angle_std = np.sqrt(elbow_angle_var)
    if lh_diff < lh_diff_mean - 3*lh_diff_std:
        if lh_diff < 0:
            result += "用户在拉臂时肘部抬的太低，击球点较低，请将手肘抬过肩膀，切勿抡大臂，谨防肩膀损伤。可以通过站立对墙扣球进行纠正，请及时反馈训练视频，不要养成错误习惯。"
        else:
            result += "用户在拉臂时肘部较低，击球点较低，可以将手肘拉起到接近耳朵位置，切勿抡大臂，谨防肩膀损伤。可以通过站立对墙扣球进行纠正，请及时反馈训练视频，不要养成错误习惯。"
    if hh_diff < hh_diff_mean - 3*hh_diff_std:
        if hh_diff < 0:
            result += "同时，用户在转体挥臂时肘部抬的太低，请将手肘抬过肩膀，体会转体时将手肘藏到耳后再转肘的感觉。"
        else:
            result += "同时，用户在转体挥臂时肘部较低，可以将手肘拉起到接近耳朵位置，体会转体时将手肘藏到耳后再转肘的感觉。"
    if hh_diff < lh_diff - lh_diff_std:
        result += "用户在挥臂时掉肘，在扣球过程中，从拉臂到转体挥臂都需要始终保持肘部不往下掉，尽量提高击球点。"
    if lh_angle > lh_angle_mean + 3*lh_angle_std or hh_angle > hh_angle_mean + 3*hh_angle_std:
        result += "用户扣球时身体姿势过于倾斜，请注意找准球的位置，不要钻球。可以通过起跳抓球的方式训练找球能力，体会够着球去打的感觉。"
    if arm_angle > arm_angle_mean + 3*arm_angle_std:
        result += "身体朝向与扣球方向不一致，请注意，如果用户是面斜打斜，扣球时尽量保证身体朝向与扣球方向一致，确保能集中发力，请调整人球关系，在正前上方击球，"
    if elbow_angle < elbow_angle_mean - 3*elbow_angle_std:
        result += "挥臂时整个手臂不够伸直，过于弯曲，扣球动作无法做完整，发力不够且打点较低。请在站立扣球时进行针对练习，建议练习时完整做完整个扣球动作，"
    if result == "":
        result += "用户扣球时挥臂动作标准，保持状态，继续训练，持续反馈！。"
    return result

def test_height_info(kpt):
    # 输入kpt是一个二维数组，第一维为关节点，第二维为坐标 (x, y, z)
    # 关节点索引定义如下：
    # 0:臀部 7：脊椎 8：胸部  14：右肩 15：右肘 16：右手腕 13:左手腕
    # 1：右髋 4：左髋
    # 提取需要的关节点坐标
    thorax = kpt[8]
    right_shoulder = kpt[14]
    right_elbow = kpt[15]
    right_hip = kpt[1]
    left_hip = kpt[4]
    hip = kpt[0]
    spine = kpt[7]
    # 计算右肘、右肩高度
    right_shoulder_height = right_shoulder[2]
    right_elbow_height = right_elbow[2]
    # 计算胸部到臀部的长度，推测身长
    thorax_to_hip_length = np.linalg.norm(thorax - hip)
    # 计算身体与竖直方向的夹角
    body_vector = spine - (left_hip + right_hip) / 2
    vertical_vector = np.array([0, -1, 0])  # y轴向上
    body_angle = np.degrees(np.arccos(np.abs(
        np.dot(body_vector, vertical_vector) / (np.linalg.norm(body_vector) * np.linalg.norm(vertical_vector)))))

    # 臂展与身高成1：1，通过计算与身高的比例，消除身高产生的差异
    shoulder_elbow_height = (right_shoulder_height - right_elbow_height) / thorax_to_hip_length
    return shoulder_elbow_height, body_angle


def test_arm_angle(kpt):
    # 输入kpt是一个二维数组，第一维为关节点，第二维为坐标 (x, y, z)
    # 关节点索引定义如下：
    # 0: 臀部 7: 腹部 8: 颈部 10:头部 14: 右肩 15: 右肘 16: 右手腕
    # 1: 右髋 4: 左髋
    # 该函数目的为返回输入帧胳膊与身体朝向面的夹角，以及手腕到手肘、手肘到肩膀之间的夹角

    right_shoulder = kpt[14]
    right_elbow = kpt[15]
    right_wrist = kpt[16]
    hip = kpt[0]
    abdomen = kpt[7]
    neck = kpt[8]
    # 计算肩膀到手腕的向量
    arm_vector = right_wrist - right_shoulder

    # 计算身体朝向面的法向量
    body_normal_vector = np.cross(abdomen - hip, neck - hip)

    # 计算向量的夹角
    dot_product = np.dot(arm_vector, body_normal_vector)
    magnitudes = np.linalg.norm(arm_vector) * np.linalg.norm(body_normal_vector)
    arm_body_angle = np.degrees(np.arcsin(1) - np.arccos(np.abs(dot_product / magnitudes)))

    # 计算手腕到手肘、手肘到肩膀之间的夹角
    wrist_to_elbow = right_wrist - right_elbow
    elbow_to_shoulder = right_elbow - right_shoulder

    dot_product_arm = np.abs(np.dot(wrist_to_elbow, elbow_to_shoulder))
    magnitudes_arm = np.linalg.norm(wrist_to_elbow) * np.linalg.norm(elbow_to_shoulder)
    elbow_angle = np.degrees(np.arccos(dot_product_arm / magnitudes_arm))

    return arm_body_angle, elbow_angle

def test_block_pose():
    return "1"