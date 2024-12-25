import numpy as np

elbow_angle_data = []
knee_angle_data = []
hand_eye_data = []
arm_angle_data = [] 
hand_track_data = []

def save_data(elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track_data):
    np.savez('ui/data.npz', 
             elbow_angle_data=elbow_angle_data, 
             knee_angle_data=knee_angle_data, 
             hand_eye_data=hand_eye_data, 
             arm_angle_data=arm_angle_data, 
             hand_track_data=hand_track_data)
    print("Data saved to data.npz")

def load_data():
    data = np.load('ui/data.npz')
    elbow_angle_data = data['elbow_angle_data']
    knee_angle_data = data['knee_angle_data']
    hand_eye_data = data['hand_eye_data']
    arm_angle_data = data['arm_angle_data']
    hand_track_data = data['hand_track_data']
    print("Data loaded from data.npz")
    return elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track_data

if __name__ == "__main__":
    save_data(elbow_angle_data, knee_angle_data, hand_eye_data, arm_angle_data, hand_track_data)
