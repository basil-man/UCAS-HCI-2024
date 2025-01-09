import numpy as np
import os
from deepsquat_data import save_data, load_data

# 定义关键点索引
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
RIGHT_SHOULDER = 11  # 右肩
RIGHT_ELBOW = 12  # 右肘
RIGHT_HAND = 13  # 右手腕
LEFT_SHOULDER = 14  # 左肩
LEFT_ELBOW = 15  # 左肘
LEFT_HAND = 16  # 左手腕

def test_squat_pose(train=False, video_path="./demo/output/square_vedio/"):
    """
    分析深蹲动作并提供反馈。
    train: 是否将数据用于训练
    video_path: 视频数据路径
    """
    result = ""

    try:
        # 加载2D和3D关键点数据
        keypoints_2d = np.load(os.path.join(video_path, "input_2D", "keypoints.npz"), allow_pickle=True)["reconstruction"]
        keypoints_3d = np.load(os.path.join(video_path, "output_3D", "output_keypoints_3d.npz"), allow_pickle=True)["reconstruction"]
    except FileNotFoundError as e:
        return f"文件加载失败: {e}"

    # 确定动作起始和结束帧
    start_idx = -1
    lowest_idx = -1
    finish_idx = -1
    min_knee_angle = 180

    for i in range(keypoints_3d.shape[0]):
        kpt = keypoints_3d[i, :, :]

        # Detect the start of squat (standing position with knees extended)
        if start_idx == -1:
            knee_angle = test_knee_angle(kpt)
            if knee_angle > 130:
                start_idx = i

        # Detect the lowest squat position (minimum knee angle)
        if start_idx != -1 and lowest_idx == -1:
            knee_angle = test_knee_angle(kpt)
            if knee_angle < min_knee_angle:
                min_knee_angle = knee_angle
                lowest_idx = i

        # Detect the end of squat (return to standing position)
        if lowest_idx != -1:
            knee_angle = test_knee_angle(kpt)
            if knee_angle > 130:
                finish_idx = i
                break

    if start_idx == -1 or lowest_idx == -1 or finish_idx == -1:
        return "深蹲动作不完整，视频录制过短，请重新录制视频，确保上传完整的深蹲动作！"
        

    lowest_position = keypoints_3d[lowest_idx, :, :]
    start_position = keypoints_3d[start_idx, :, :]
    finish_position = keypoints_3d[finish_idx, :, :]

    height = test_height(start_position)

    squat_depth = test_squat_depth(lowest_position, start_position)
    end_depth = test_end_depth(finish_position, start_position)
    back_angle = test_back_angle(lowest_position)
    foot_width = test_foot_width(start_position)
    knee_width = test_knee_width(lowest_position)
    shoulder_width = test_shoulder_width(start_position)
    foot_knee_alignment = test_foot_knee_alignment(start_position, lowest_position)
    head_back_alignment = test_head_back_alignment(lowest_position)

    # 加载保存的数据
    knee_angle_data, squat_depth_data, squat_end_data, back_angle_data, foot_knee_data, foot_shoulder_width, knee_toe_data, hip_heel_data, head_angle_data, center_of_gravity_track = load_data()

    if train:
        # 保存训练数据
        knee_angle_data = np.append(knee_angle_data, min_knee_angle)
        squat_depth_data = np.append(squat_depth_data, squat_depth)
        squat_end_data = np.append(squat_end_data, end_depth)
        back_angle_data = np.append(back_angle_data, back_angle)
        save_data(knee_angle_data, squat_depth_data, squat_end_data, back_angle_data, foot_knee_data, foot_shoulder_width, knee_toe_data, hip_heel_data, head_angle_data, center_of_gravity_track)
        result = "训练数据已保存"
        with open(os.path.join(video_path, "result.txt"), "w") as f:
            f.write(result)
        return result

    # 计算均值和标准差
    knee_angle_mean = np.mean(knee_angle_data)
    squat_depth_mean = np.mean(squat_depth_data)
    back_angle_mean = np.mean(back_angle_data)

    knee_angle_std = np.std(knee_angle_data)
    squat_depth_std = np.std(squat_depth_data)
    back_angle_std = np.std(back_angle_data)

    # 评估深蹲动作
    if min_knee_angle < knee_angle_mean - 3 * knee_angle_std:
        result += "用户在深蹲时膝盖角度过小，可能会导致膝盖过度弯曲，增加膝关节的压力，可能造成以下伤害：膝关节前方的软骨（如髌骨软骨）会承受过大的剪切力，增加软骨磨损的风险；韧带和肌腱的张力异常，容易劳损。请注意控制膝盖的弯曲角度，在90度以上，避免超过标准范围。\n"

    if squat_depth < 0.3 * height:
        result += "用户深蹲的下蹲深度不足，可能会导致目标肌群（如股四头肌、臀大肌）未充分激活。这样的危害有：股四头肌和臀部肌群的力量发展不足，可能导致身体不平衡；大腿后侧肌群（腘绳肌）未能适当拉伸，可能导致柔韧性下降。请确保深蹲时臀部下降到膝盖以下的位置。\n"
    elif squat_depth > 0.5 * height:
        result += "用户深蹲的深度过大，可能会导致腰椎和髋关节过度屈曲，增加下背部的压力，可能导致腰椎间盘突出等疾病。请控制下蹲深度在合理范围内，避免增加受伤风险。\n"

    if end_depth > 0.2 * height:
        result += "用户在深蹲结束时未能完全站立，可能会导致下背部和膝盖过度压力，增加腰椎和膝关节的受伤风险。这样的危害有：腰椎间盘受压，可能导致腰椎间盘突出；膝关节承受额外压力，可能导致膝盖软骨磨损。请确保深蹲结束时完全站立，避免过度压力。\n"

    if back_angle < back_angle_mean - 3 * back_angle_std:
        result += "用户在深蹲时背部前倾角度过小，可能导致动作僵硬，影响下肢力量的传递。这样的危害有：动作协调性降低，力量无法均匀分配到股四头肌和臀大肌，增加肌肉劳损风险；下背部肌群未能充分参与支撑，可能导致身体核心稳定性降低。请适当前倾上半身，在10-30度之间，保持动作流畅。\n"
    elif back_angle > back_angle_mean + 3 * back_angle_std:
        result += "用户在深蹲时背部前倾角度过大，可能会导致下背部压力过大，很可能导致：脊柱负荷增加，使得脊柱过度伸展或椎间盘突出；背部肌肉（如竖脊肌）过度紧张，可能引发急性或慢性肌肉拉伤；力量分配不均导致其他部位（如膝盖或髋关节）代偿性动作，进一步增加受伤风险。请保持脊柱中立，在10-30度之间，避免过度前倾。\n"

    if foot_width < 0.8 * shoulder_width:
        result += "用户的双脚间距太窄，这样的危害有：深蹲时膝盖容易向内塌陷，增加内侧副韧带（MCL）的受伤风险；下肢的力量无法均匀分布，可能导致过度依赖膝盖关节，减少髋部的参与。建议双脚与肩同宽，确保动作的稳定性和安全性。\n"
    elif foot_width > 1.2 * shoulder_width:
        result += "用户的双脚间距太宽，这样的危害有：髋关节的活动范围受到限制，可能增加髋关节压力；动作重心不稳定，容易导致背部或膝盖代偿。建议双脚与肩同宽，确保动作的稳定性和安全性。\n"

    if knee_width < foot_width:
        result += "用户的膝盖间距小于双脚间距，可能导致膝盖内扣（俗称“X腿”现象）。这样的危害包括：膝盖关节内侧的韧带和软骨承受过大的压力，长期可能导致韧带拉伤或软骨磨损；膝盖内扣可能影响下肢的力线，增加髋关节和脚踝的代偿性压力。请保持膝盖和脚尖方向一致，膝盖间距适当。\n"

    if not foot_knee_alignment:
        result += "用户深蹲时膝盖与脚尖未对齐，可能导致膝关节承受额外压力。这种错误可能导致：膝盖承受异常的旋转力，增加髌骨和关节软骨的磨损风险；下肢力线不正，可能导致脚踝、髋关节等部位的代偿性压力。请调整膝盖方向，与脚尖方向一致。\n"

    if not head_back_alignment:
        result += "用户的头部和背部未保持直线，可能影响动作稳定性。这样的危害包括：低头可能导致颈椎承受额外压力，增加颈部肌肉劳损的风险；仰头可能导致脊柱上部过度伸展，影响脊柱整体的稳定性。请保持头部和背部在同一直线上，避免低头或过度仰头。\n"

    if result == "":
        result += "用户深蹲动作标准，保持状态，继续训练，持续反馈！\n"
        result += "良好的深蹲动作能够有效激活目标肌群，增强核心稳定性和下肢力量。继续保持并优化训练细节。\n"

    with open(os.path.join(video_path, "result.txt"), "w") as f:
        f.write(result)
    print(result)
    return result

# 定义辅助功能
def test_height(keypoints_3d):
    hip, head = keypoints_3d[HIP], keypoints_3d[HEAD]
    return head[1] - hip[1]

def test_knee_angle(kpt):
    hip, knee, ankle = kpt[LEFT_HIP], kpt[LEFT_KNEE], kpt[LEFT_FOOT]
    vec_thigh, vec_shin = hip - knee, ankle - knee
    cos_angle = np.dot(vec_thigh, vec_shin) / (np.linalg.norm(vec_thigh) * np.linalg.norm(vec_shin))
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    print(angle, hip, knee, ankle)
    return angle

def test_squat_depth(lowest_position, start_position):
    return start_position[HIP, 1] - lowest_position[HIP, 1]

def test_end_depth(finish_position, start_position):
    return finish_position[HIP, 1] - start_position[HIP, 1]

def test_back_angle(lowest_position):
    hip, neck = lowest_position[HIP], lowest_position[NECK]
    vec_back, vec_vertical = neck - hip, np.array([0, 1, 0])
    cos_angle = np.dot(vec_back, vec_vertical) / (np.linalg.norm(vec_back) * np.linalg.norm(vec_vertical))
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def test_foot_width(start_position):
    return np.linalg.norm(start_position[LEFT_FOOT] - start_position[RIGHT_FOOT])

def test_knee_width(lowest_position):
    return np.linalg.norm(lowest_position[LEFT_KNEE] - lowest_position[RIGHT_KNEE])

def test_shoulder_width(start_position):
    return np.linalg.norm(start_position[LEFT_SHOULDER] - start_position[RIGHT_SHOULDER])

def test_foot_knee_alignment(start_position, lowest_position):
    left_foot, right_foot = start_position[LEFT_FOOT], start_position[RIGHT_FOOT]
    left_knee, right_knee = lowest_position[LEFT_KNEE], lowest_position[RIGHT_KNEE]
    return (np.linalg.norm(left_foot - left_knee) < np.linalg.norm(left_foot - right_knee) and
            np.linalg.norm(right_foot - right_knee) < np.linalg.norm(right_foot - left_knee))

def test_head_back_alignment(lowest_position):
    head, neck, hip = lowest_position[HEAD], lowest_position[NECK], lowest_position[HIP]
    vec_back = neck - hip
    vec_head = head - neck
    cos_angle = np.dot(vec_back, vec_head) / (np.linalg.norm(vec_back) * np.linalg.norm(vec_head))
    angle = np.arccos(cos_angle) * (180.0 / np.pi)
    return angle < 10  # 允许的误差范围


if __name__ == "__main__":
    test_squat_pose()  # 测试数据