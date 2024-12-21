import numpy as np

elbow_angle_data = np.array([90, 80, 95, 100, 105])
knee_angle_data = np.array([100, 120, 125, 95, 110, 115, 105])
hand_eye_data = np.array([0.25, 0.3, 0.2, 0.15, 0.4, 0.15])
arm_angle_data = np.array([120, 130, 150, 160, 140, 140, 130])
hand_track = np.array([0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.2, 0.15])

def save_data(elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track):
    np.savez('ui/data.npz', 
             elbow_angle_data=elbow_angle_data, 
             knee_angle_data=knee_angle_data, 
             hand_eye_data=hand_eye_data, 
             arm_angle_data=arm_angle_data, 
             hand_track=hand_track)
    print("Data saved to data.npz")

def load_data():
    data = np.load('ui/data.npz')
    elbow_angle_data = data['elbow_angle_data']
    knee_angle_data = data['knee_angle_data']
    hand_eye_data = data['hand_eye_data']
    arm_angle_data = data['arm_angle_data']
    hand_track = data['hand_track']
    print("Data loaded from data.npz")
    return elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track
