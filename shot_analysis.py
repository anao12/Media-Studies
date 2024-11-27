import cv2
import os
import numpy as np
from datetime import datetime
import subprocess  # For video format conversion using ffmpeg
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Configuration Parameters
FACE_MIN_SIZE = (80, 80)  # Minimum face size for detection
WIDE_SHOT_MAX_FACE_RATIO = 0.05  # If face area is less than 5% of the frame, it's a wide shot
CLOSE_UP_RATIO_THRESHOLD = 0.20  # Face area to frame area ratio for a close-up
ANALYSIS_FREQUENCY_SECONDS = 0.5  # Analyze every 0.5 seconds of video

# Set the path to the models directory
model_dir = "YOUR_DIRECTORY"
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")


# Load the DNN model files
prototxt_path = os.path.join(model_dir, "deploy.prototxt")
model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Verify model files exist
if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    raise FileNotFoundError("Model files not found. Please check paths.")

# Load the pre-trained MobileNet-SSD model for face detection
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Convert .mov or .MOV files to .mp4 using ffmpeg
def convert_video_format(video_path):
    video_name, video_ext = os.path.splitext(video_path)
    if video_ext.lower() in [".mov", ".MOV"]:
        mp4_path = f"{video_name}.mp4"
        logging.info(f"Converting {video_path} to {mp4_path}...")
        try:
            subprocess.run(["ffmpeg", "-i", video_path, mp4_path], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting video: {e}")
            return None
        return mp4_path
    return video_path  # If already in .mp4 format, no need to convert

# Function to classify shot types: Close-Up and Wide Shot
def classify_shot(frame):
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_width * frame_height

    # Prepare the frame for the DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # Confidence threshold for detection
            box = detections[0, 0, i, 3:7] * np.array([frame_width, frame_height, frame_width, frame_height])
            (x, y, x1, y1) = box.astype("int")
            face_width = x1 - x
            face_height = y1 - y
            if face_width > FACE_MIN_SIZE[0] and face_height > FACE_MIN_SIZE[1]:  # Ensure minimum face size
                faces.append((x, y, face_width, face_height))

    # If no faces or very small faces are detected, classify as Wide Shot (WS)
    if len(faces) == 0 or all((w * h) / frame_area < WIDE_SHOT_MAX_FACE_RATIO for (x, y, w, h) in faces):
        return "WS"

    # If there is one face and it's large enough, classify as Close-Up
    if len(faces) == 1:
        (x, y, w, h) = faces[0]
        face_area = w * h
        face_to_frame_ratio = face_area / frame_area

        # Close-Up: If the face occupies more than 20% of the frame area
        if face_to_frame_ratio > CLOSE_UP_RATIO_THRESHOLD:
            return "CU"

    return None  # Neither type detected

# Save frame to image file
def save_frame(frame, second, shot_type, video_filename):
    output_dir = os.path.join(os.path.dirname(video_filename), "video_stills")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{base_name}_at_{second:.1f}_sec_{shot_type}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    logging.info(f"Saved frame: {filename}")
    return filename

# Function to create a collage of detected frames
def create_collage(image_files, output_dir, video_name):
    if len(image_files) == 0:
        logging.warning("No images to create a collage.")
        return

    # Load images and resize them to the same size
    images = [cv2.imread(img) for img in image_files]
    images = [cv2.resize(img, (300, 200)) for img in images]  # Resize for uniformity

    cols = 2  # Number of images per row
    rows = (len(images) + cols - 1) // cols  # Number of rows needed
    h, w = images[0].shape[:2]  # Height and width of each image
    collage = np.zeros((h * rows, w * cols, 3), dtype=np.uint8)  # Create an empty collage

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        collage[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    collage_filename = os.path.join(output_dir, f"{video_name}_collage_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
    cv2.imwrite(collage_filename, collage)
    logging.info(f"Collage saved as {collage_filename}")

# Function to analyze a single video
def analyze_video(video_path):
    logging.info(f"Analyzing video: {video_path}")

    # Convert video if necessary and update video_path
    video_path = convert_video_format(video_path)
    if video_path is None:
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    analysis_frequency = int(fps * ANALYSIS_FREQUENCY_SECONDS)
    duration_seconds = min(30, total_frames // fps)  # Limit to 30 seconds or video duration

    saved_closeup = False
    saved_wideshot = False

    # List to store the filenames of saved frames for collage creation
    saved_frames = []

    for frame_idx in range(0, total_frames, analysis_frequency):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_time_in_seconds = frame_idx / fps
        frame = cv2.resize(frame, (640, 360))
        shot_type = classify_shot(frame)

        if shot_type == "CU" and not saved_closeup:
            filename = save_frame(frame, current_time_in_seconds, shot_type, video_path)
            saved_frames.append(filename)
            saved_closeup = True
        elif shot_type == "WS" and not saved_wideshot:
            filename = save_frame(frame, current_time_in_seconds, shot_type, video_path)
            saved_frames.append(filename)
            saved_wideshot = True

        if saved_closeup and saved_wideshot:
            break

    cap.release()

    # Create a collage of the saved frames
    output_dir = os.path.join(os.path.dirname(video_path), "video_stills")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    create_collage(saved_frames, output_dir, video_name)

# Function to process multiple videos
def analyze_multiple_videos(video_paths):
    for video_path in video_paths:
        analyze_video(video_path)

# Example usage with multiple video files
video_paths = [
    "YOUR_VIDEO_PATHS"
]

analyze_multiple_videos(video_paths)



