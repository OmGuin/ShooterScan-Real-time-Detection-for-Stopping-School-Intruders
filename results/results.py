from moviepy.video.io.VideoFileClip import VideoFileClip

# Load the video
input_video = "VID_20241202_234744.mp4"  # Replace with the path to your video
output_folder = "clips/"  # Folder to save the clips

# Create the output folder if it doesn't exist
import os
os.makedirs(output_folder, exist_ok=True)

# Load the video file
video = VideoFileClip(input_video)

# Get the video duration
duration = int(video.duration)

# Split the video into 1-second clips
for start_time in range(0, duration):
    end_time = start_time + 1
    clip = video.subclipped(start_time, end_time)
    output_file = os.path.join(output_folder, f"clip_{start_time+1}.mp4")
    clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

print("All clips have been successfully saved!")