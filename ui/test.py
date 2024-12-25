import numpy as np
from benchpress_data import save_data, load_data

def add_new_data():
    new_elbow_angle = float(input("Enter new elbow angle: "))
    new_knee_angle = float(input("Enter new knee angle: "))
    new_hand_eye = float(input("Enter new hand-eye data: "))
    new_arm_angle = float(input("Enter new arm angle: "))
    new_hand_track = float(input("Enter new hand track data: "))

    elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track = load_data() 

    elbow_angle_data = np.append(elbow_angle_data, new_elbow_angle)
    knee_angle_data = np.append(knee_angle_data, new_knee_angle)
    hand_eye_data = np.append(hand_eye_data, new_hand_eye)
    arm_angle_data = np.append(arm_angle_data, new_arm_angle)
    hand_track = np.append(hand_track, new_hand_track)

    print("Updated elbow_angle_data:", elbow_angle_data)
    print("Updated knee_angle_data:", knee_angle_data)
    print("Updated hand_eye_data:", hand_eye_data)
    print("Updated arm_angle_data:", arm_angle_data)
    print("Updated hand_track:", hand_track)

    save_data(elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track)

# 调用函数以添加新数据
add_new_data()