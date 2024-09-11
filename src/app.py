from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from PIL import Image
import io
from ultralytics import YOLO
import mediapipe as mp
from flask_cors import CORS
import time
import torch
import drawing  # Assuming custom drawing functions in drawing.py

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize YOLOv8 Model for person detection
model = YOLO("yolov8n.pt")
if torch.cuda.is_available():
    model.to('cuda')
else:
    print("Warning: CUDA not available. Using CPU instead.")
    model.to('cpu')

# MediaPipe Pose Initialization
mp_pose = mp.solutions.pose.Pose()

# Global variables for additional features
face_blur_enabled = False

# Frame processing function
def process_frame(frame):
    global face_blur_enabled

    start_time = time.time()  # Start frame time

    # Flip the frame horizontally (mirror effect)
    frame = cv2.flip(frame, 1)

    # Run YOLOv8 object detection
    results = model(frame)
    boxes = results[0].boxes.cpu().numpy()  # Extract bounding boxes

    # Get the center of the frame
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2

    # Find the box closest to the center of the frame
    closest_box = None
    min_distance = float('inf')

    for box in boxes:
        if box.cls == 0:  # YOLO class 0 corresponds to 'person'
            # Get the center of the bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2

            # Calculate the Euclidean distance from the center of the frame
            distance = ((box_center_x - frame_center_x) ** 2 + (box_center_y - frame_center_y) ** 2) ** 0.5

            # Check if this box is closer to the center
            if distance < min_distance:
                min_distance = distance
                closest_box = (x1, y1, x2, y2)

    # Process only the closest box
    if closest_box:
        x1, y1, x2, y2 = closest_box

        # Crop the detected person from the frame
        person_frame = frame[y1:y2, x1:x2]

        # Convert cropped person frame to RGB for MediaPipe processing
        person_rgb = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)

        # Apply MediaPipe Pose Estimation
        results_pose = mp_pose.process(person_rgb)

        # Get keypoints as a list of (x, y) pairs
        if results_pose.pose_landmarks:
            keypoints = [(lm.x, lm.y) for lm in results_pose.pose_landmarks.landmark]

            # Draw keypoints and connections using the custom drawing utilities
            drawing.draw_keypoints_and_connections(person_frame, keypoints)

            # Apply face blur if enabled
            if face_blur_enabled:
                drawing.blur_face(person_frame, keypoints)

        # Place the processed person frame back into the original frame
        frame[y1:y2, x1:x2] = person_frame

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate and print total frame processing time
    end_time = time.time()
    frame_time = end_time - start_time
    print(f"Total frame time: {frame_time:.2f} seconds")

    return frame_rgb

# Handle incoming WebSocket frames
@socketio.on('frame')
def handle_frame(data):
    # Decode base64 image
    frame_data = data.split(',')[1]
    frame_bytes = base64.b64decode(frame_data)
    image = Image.open(io.BytesIO(frame_bytes))

    # Convert image to OpenCV format
    frame = np.array(image)

    # Process frame with YOLO and Pose
    processed_frame = process_frame(frame)

    # Encode processed frame back to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send processed frame back to client
    emit('processed_frame', 'data:image/jpeg;base64,' + processed_frame_base64)

# Route to set face blur
@app.route('/set_face_blur/<string:state>', methods=['GET'])
def set_face_blur(state):
    global face_blur_enabled
    face_blur_enabled = (state == 'enabled')  # Enable or disable face blur
    print(f'Face blur {state}')
    return '', 200

# Serve the client-side HTML
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
