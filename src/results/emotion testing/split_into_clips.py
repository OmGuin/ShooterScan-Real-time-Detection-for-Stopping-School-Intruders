from moviepy.video.io.VideoFileClip import VideoFileClip

input_video = "../VID_20241202_234744.mp4"
output_folder = "../clips/" 

import os
os.makedirs(output_folder, exist_ok=True)

video = VideoFileClip(input_video)

duration = int(video.duration)

for start_time in range(0, duration):
    end_time = start_time + 1
    clip = video.subclipped(start_time, end_time)
    output_file = os.path.join(output_folder, f"clip_{start_time+1}.mp4")
    clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

print("All clips have been successfully saved!")