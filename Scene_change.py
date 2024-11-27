import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Define video path
video_path = "video-path"

# Check if the file exists
if not os.path.exists(video_path):
    print(f"Error: File '{video_path}' not found.")
    exit()

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

print(f"Video properties: FPS={fps}, Total Frames={total_frames}")

# Initialize variables for frame differences
frame_diffs = []
scene_changes = []
prev_frame = None

# Process the video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference with the previous frame
    if prev_frame is not None:
        diff = cv2.absdiff(gray_frame, prev_frame)
        diff_sum = np.sum(diff)  # Sum of pixel differences
        frame_diffs.append(diff_sum)

    # Update the previous frame
    prev_frame = gray_frame

cap.release()

# Normalize frame differences
frame_diffs = np.array(frame_diffs)
normalized_diffs = (frame_diffs - np.min(frame_diffs)) / (np.max(frame_diffs) - np.min(frame_diffs))

# Set a threshold for detecting scene changes
threshold = 0.5  # You can adjust this based on sensitivity
scene_changes = np.where(normalized_diffs > threshold)[0]

# Convert frame numbers to timestamps
scene_change_times = [frame / fps for frame in scene_changes]

print("Detected Scene Changes:")
for frame_idx, time in zip(scene_changes, scene_change_times):
    print(f"Frame {frame_idx}, Time: {time:.2f} seconds")

# Plot the frame differences with detected scene changes
plt.figure(figsize=(10, 5))
plt.plot(normalized_diffs, label="Frame Differences")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold")
plt.title("Scene Change Detection")
plt.xlabel("Frame Number")
plt.ylabel("Normalized Difference")

# Annotate scene changes on the graph
for frame_idx, time in zip(scene_changes, scene_change_times):
    plt.text(frame_idx, normalized_diffs[frame_idx], f"{time:.2f}s", color='blue', fontsize=8)

plt.legend()
plt.show()

# Save results to a text file
output_file = "scene_changes_with_times.txt"
with open(output_file, "w") as f:
    for frame_idx, time in zip(scene_changes, scene_change_times):
        f.write(f"Frame: {frame_idx}, Time: {time:.2f} seconds\n")

print(f"Scene change information saved to '{output_file}'.")

# Optional: Annotate scene changes in the video (if needed)
output_video_path = "annotated_video.mp4"
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

current_frame = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if current_frame in scene_changes:
        cv2.putText(frame, "Scene Change", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    out.write(frame)
    current_frame += 1

cap.release()
out.release()
print(f"Annotated video saved as '{output_video_path}'.")

