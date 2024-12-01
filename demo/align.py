import numpy as np
import cv2
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import moviepy.editor as mpy


def load_keypoints(file_path):
    keypoints_3d = np.load(file_path, allow_pickle=True)["reconstruction"]
    return keypoints_3d


def find_best_match(kp_short, kp_long):
    # 将关键点展平为一维向量
    kp_short_flat = kp_short.reshape(kp_short.shape[0], -1)
    kp_long_flat = kp_long.reshape(kp_long.shape[0], -1)

    min_distance = float("inf")
    best_start = 0
    len_short = kp_short_flat.shape[0]

    # 遍历 kp_long，寻找与 kp_short 最相似的子序列
    for i in range(kp_long_flat.shape[0] - len_short + 1):
        kp_long_segment = kp_long_flat[i : i + len_short]
        distance, _ = fastdtw(kp_short_flat, kp_long_segment, dist=euclidean)
        if distance < min_distance:
            min_distance = distance
            best_start = i

    # 截取最佳匹配段
    kp_long_best = kp_long[best_start : best_start + len_short]
    return kp_short, kp_long_best, best_start


def temporal_alignment_dtw(kp1, kp2):
    # 将每帧关键点展平为一维向量
    kp1_flat = kp1.reshape(kp1.shape[0], -1)
    kp2_flat = kp2.reshape(kp2.shape[0], -1)

    # 计算 DTW 路径
    distance, path = fastdtw(kp1_flat, kp2_flat, dist=euclidean)

    # 根据路径对齐关键点
    kp1_aligned = []
    kp2_aligned = []
    for i, j in path:
        kp1_aligned.append(kp1[i])
        kp2_aligned.append(kp2[j])

    return np.array(kp1_aligned), np.array(kp2_aligned)


def spatial_alignment(kp1, kp2):
    center1 = np.mean(kp1, axis=1, keepdims=True)
    center2 = np.mean(kp2, axis=1, keepdims=True)
    kp2_aligned = kp2 - center2 + center1
    return kp1, kp2_aligned


# def synchronize_videos(video_path1, video_path2, start_time, duration, output_path):
#     clip1 = mpy.VideoFileClip(video_path1).subclip(0, duration)
#     clip2 = mpy.VideoFileClip(video_path2).subclip(start_time, start_time + duration)


#     synchronized = mpy.clips_array([[clip1, clip2]])
#     synchronized.write_videofile(output_path)
def synchronize_videos(video_path1, video_path2, start_time, duration, output_path):
    clip1 = mpy.VideoFileClip(video_path1).subclip(0, duration)
    clip2 = mpy.VideoFileClip(video_path2).subclip(start_time, start_time + duration)

    # 计算速度调整因子
    duration1 = clip1.duration
    duration2 = clip2.duration
    speed_factor = duration1 / duration2

    # 调整第二个视频的速度
    clip2 = clip2.fx(mpy.vfx.speedx, speed_factor)

    synchronized = mpy.clips_array([[clip1, clip2]])
    synchronized.write_videofile(output_path)


def align_videos(npz1, npz2, video1, video2, output_video, fps=30):
    kp1 = load_keypoints(npz1)
    kp2 = load_keypoints(npz2)

    # 找到最佳匹配段
    kp1_matched, kp2_matched, best_start = find_best_match(kp1, kp2)

    # 计算匹配段的开始时间
    start_time = best_start / fps
    duration = len(kp1_matched) / fps

    # DTW 微调
    kp1_aligned, kp2_aligned = temporal_alignment_dtw(kp1_matched, kp2_matched)
    kp1_aligned, kp2_aligned = spatial_alignment(kp1_aligned, kp2_aligned)

    # 同步视频
    synchronize_videos(video1, video2, start_time, duration, output_video)


if __name__ == "__main__":
    align_videos(
        "demo/output/deadlift/output_3D/output_keypoints_3d.npz",
        "demo/output/my_deadlift_3/output_3D/output_keypoints_3d.npz",
        "demo/video/deadlift.mp4",
        "demo/video/my_deadlift_3.mp4",
        "align/comparison_aligned.mp4",
    )
