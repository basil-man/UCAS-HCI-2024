import numpy as np

def save_data(
    knee_angle_data,
    squat_depth_data,
    squat_end_data,
    back_angle_data,
    foot_knee_data,
    foot_shoulder_width,
    knee_toe_data,
    hip_heel_data,
    head_angle_data,
    center_of_gravity_track,
):
    np.savez(
        "ui/deepsquat_data.npz",
        knee_angle_data=knee_angle_data,
        squat_depth_data=squat_depth_data,
        squat_end_data=squat_end_data,
        back_angle_data=back_angle_data,
        foot_knee_data=foot_knee_data,
        foot_shoulder_width=foot_shoulder_width,
        knee_toe_data=knee_toe_data,
        hip_heel_data=hip_heel_data,
        head_angle_data=head_angle_data,
        center_of_gravity_track=center_of_gravity_track,
    )
    print("Deep Squat data saved to deepsquat_data.npz")


def load_data():
    data = np.load("ui/deepsquat_data.npz")
    knee_angle_data = data["knee_angle_data"]
    squat_depth_data = data["squat_depth_data"]
    squat_end_data = data["squat_end_data"]
    back_angle_data = data["back_angle_data"]
    foot_knee_data = data["foot_knee_data"]
    foot_shoulder_width = data["foot_shoulder_width"]
    knee_toe_data = data["knee_toe_data"]
    hip_heel_data = data["hip_heel_data"]
    head_angle_data = data["head_angle_data"]
    center_of_gravity_track = data["center_of_gravity_track"]
    return (
        knee_angle_data,
        squat_depth_data,
        squat_end_data,
        back_angle_data,
        foot_knee_data,
        foot_shoulder_width,
        knee_toe_data,
        hip_heel_data,
        head_angle_data,
        center_of_gravity_track,
    )

if __name__ == "__main__":
    knee_angle_data = []
    squat_depth_data = []
    squat_end_data = []
    back_angle_data = []
    foot_knee_data = []
    foot_shoulder_width = []
    knee_toe_data = []
    hip_heel_data = []
    head_angle_data = []
    center_of_gravity_track = []
    save_data(
        knee_angle_data,
        squat_depth_data,
        squat_end_data,
        back_angle_data,
        foot_knee_data,
        foot_shoulder_width,
        knee_toe_data,
        hip_heel_data,
        head_angle_data,
        center_of_gravity_track,
    )
    print(load_data())

