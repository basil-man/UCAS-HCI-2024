from moviepy.editor import VideoFileClip


def make_square(video_path, output_path):
    clip = VideoFileClip(video_path)
    width, height = clip.size
    if width == height:
        clip.write_videofile(output_path, codec="libx264")
    else:
        size = max(width, height)
        clip = clip.on_color(size=(size, size), color=(0, 0, 0), pos="center")
        clip.write_videofile(output_path, codec="libx264")
