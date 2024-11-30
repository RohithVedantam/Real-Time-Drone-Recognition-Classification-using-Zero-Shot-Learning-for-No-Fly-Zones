from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import shutil
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

app = Flask(__name__)
static_folder = r"Z:\SEM-7\Capstone-2\ZSL_project\static"
frames_folder = r"Z:\SEM-7\Capstone-2\ZSL_project\static\detected_object_frame"
os.makedirs(static_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

# Load YOLOv3 model
yolo_net = cv2.dnn.readNetFromDarknet(r"Z:\SEM-7\Capstone-2\ZSL_project\yolov3.cfg", r"Z:\SEM-7\Capstone-2\ZSL_project\yolov3.weights")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load pre-trained CLIP model and processor
model_name = "openai/clip-vit-base-patch16"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Function to load the embeddings and labels from the saved files
def load_embeddings_and_labels(embeddings_path, labels_path):
    embeddings = np.load(embeddings_path, allow_pickle=True).item()
    labels = np.load(labels_path, allow_pickle=True).item()
    return embeddings, labels

# Function to classify an image using pre-loaded embeddings and cosine similarity
def classify_with_loaded_embeddings(image_path, embeddings, labels, threshold=0.2):
    # Load image
    image = Image.open(image_path)
    image_input = clip_processor(images=image, return_tensors="pt")

    # Extract image embedding using CLIP model
    image_embedding = clip_model.get_image_features(**image_input)

    # Normalize the image embedding
    image_embedding = image_embedding / image_embedding.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = cosine_similarity(image_embedding.detach().numpy(), np.array(list(embeddings.values())))

    # Find the most similar embedding
    best_match_idx = np.argmax(similarities)
    best_match_similarity = similarities[0][best_match_idx]

    if best_match_similarity >= threshold:
        best_match_image_name = list(embeddings.keys())[best_match_idx]
        predicted_label = labels[best_match_image_name]
    else:
        predicted_label = "Unknown"

    return predicted_label

# Paths to embeddings and labels
embeddings_path = r"Z:\SEM-7\Capstone-2\ZSL_project\Clip_image_embeddings_embeddings.npy"
labels_path = r"Z:\SEM-7\Capstone-2\ZSL_project\Clip_image_embeddings_labels.npy"

# Load embeddings and labels
embeddings, labels = load_embeddings_and_labels(embeddings_path, labels_path)

# Function to process video, detect objects using YOLOv3, and save relevant frames without drawing boxes
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = 2  # 2 fps
    frame_count = 0
    saved_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frames at specified frame rate
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:
            frame_path = os.path.join(frames_folder, f"frame_{frame_count}.jpg")

            # Detect objects in the frame
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            yolo_net.setInput(blob)
            outs = yolo_net.forward(output_layers)

            # Post-processing to extract detected objects
            boxes = []
            confidences = []
            class_ids = []
            height, width, _ = frame.shape
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Save the frame as-is
            if len(indices) > 0:
                cv2.imwrite(frame_path, frame)
                saved_frames.append(frame_path)

        frame_count += 1

    cap.release()
    return saved_frames

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_file = request.files.get("video_file")
        if video_file:
            video_path = os.path.join(static_folder, video_file.filename)
            video_file.save(video_path)

            # Clear the frames folder before processing
            for filename in os.listdir(frames_folder):
                file_path = os.path.join(frames_folder, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")

            try:
                saved_frames = process_video(video_path)

                if not saved_frames:
                    return render_template("Z:\SEM-7\Capstone-2\ZSL_project\index.html", result="No objects detected in the video.")

                classifications = [classify_with_loaded_embeddings(frame, embeddings, labels) for frame in saved_frames]

                result_count = Counter(classifications)
                final_result = "Drone Detected" if result_count.get("drone", 0) >= result_count.get("bird", 0) else "Bird Detected"

                display_frame = saved_frames[3] if len(saved_frames) >= 4 else saved_frames[-1]

                for frame in saved_frames:
                    if frame != display_frame:
                        os.remove(frame)

                return redirect(url_for('result', result=final_result, image_path=display_frame))

            except Exception as e:
                return render_template("Z:\SEM-7\Capstone-2\ZSL_project\index.html", error=f"An error occurred: {str(e)}")

    return render_template("Z:\SEM-7\Capstone-2\ZSL_project\index.html")

@app.route("/result")
def result():
    result = request.args.get('result')
    image_path = request.args.get('image_path')

    # Load the final frame
    frame = cv2.imread(image_path)
    height, width, _ = frame.shape

    # Process the frame with YOLOv3 again to detect objects and draw bounding boxes
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(output_layers)

    # Post-processing to extract detected objects
    boxes = []
    confidences = []
    class_ids = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Only consider detections with confidence > 0.5
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # If any object is detected, draw a bounding box on the final frame
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Save the final frame with the bounding box drawn around the detected object
    final_image_path = os.path.join(frames_folder, "final_frame_with_box.jpg")
    cv2.imwrite(final_image_path, frame)

    return render_template("Z:\SEM-7\Capstone-2\ZSL_project\result.html", result=result, image_path=final_image_path)

if __name__ == "__main__":
    app.run(debug=True)
