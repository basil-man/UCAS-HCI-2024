import numpy as np


def save_data(
    feet_factor,
    start_leg_barbell_distance,
    heighest_leg_barbell_distance,
    finish_leg_barbell_distance,
    start_butt_navel_neck_dot_product,
    heighest_butt_navel_neck_dot_product,
    finish_butt_navel_neck_dot_product,
    start_head_neck_navel_dot_product,
    heighest_head_neck_navel_dot_product,
    finish_head_neck_navel_dot_product,
    arm_angle,
):
    np.savez(
        "ui/deadlift_data.npz",
        feet_factor=feet_factor,
        start_leg_barbell_distance=start_leg_barbell_distance,
        heighest_leg_barbell_distance=heighest_leg_barbell_distance,
        finish_leg_barbell_distance=finish_leg_barbell_distance,
        start_butt_navel_neck_dot_product=start_butt_navel_neck_dot_product,
        heighest_butt_navel_neck_dot_product=heighest_butt_navel_neck_dot_product,
        finish_butt_navel_neck_dot_product=finish_butt_navel_neck_dot_product,
        start_head_neck_navel_dot_product=start_head_neck_navel_dot_product,
        heighest_head_neck_navel_dot_product=heighest_head_neck_navel_dot_product,
        finish_head_neck_navel_dot_product=finish_head_neck_navel_dot_product,
        arm_angle=arm_angle,
    )
    print("Data saved to deadlift_data.npz")


def load_data():
    data = np.load("ui/deadlift_data.npz")
    feet_factor = data["feet_factor"]
    start_leg_barbell_distance = data["start_leg_barbell_distance"]
    heighest_leg_barbell_distance = data["heighest_leg_barbell_distance"]
    finish_leg_barbell_distance = data["finish_leg_barbell_distance"]
    start_butt_navel_neck_dot_product = data["start_butt_navel_neck_dot_product"]
    heighest_butt_navel_neck_dot_product = data["heighest_butt_navel_neck_dot_product"]
    finish_butt_navel_neck_dot_product = data["finish_butt_navel_neck_dot_product"]
    start_head_neck_navel_dot_product = data["start_head_neck_navel_dot_product"]
    heighest_head_neck_navel_dot_product = data["heighest_head_neck_navel_dot_product"]
    finish_head_neck_navel_dot_product = data["finish_head_neck_navel_dot_product"]
    arm_angle = data["arm_angle"]
    return (
        feet_factor,
        start_leg_barbell_distance,
        heighest_leg_barbell_distance,
        finish_leg_barbell_distance,
        start_butt_navel_neck_dot_product,
        heighest_butt_navel_neck_dot_product,
        finish_butt_navel_neck_dot_product,
        start_head_neck_navel_dot_product,
        heighest_head_neck_navel_dot_product,
        finish_head_neck_navel_dot_product,
        arm_angle,
    )
