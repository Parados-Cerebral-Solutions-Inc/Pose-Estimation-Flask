from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
from flask_cors import CORS
import os
import drawing  # Import custom drawing functions from drawing.py
import mediapipe as mp
import time
import torch

print(torch.cuda.is_available())

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
print("CORS enabled")

# Default camera index
camera_index = 0
face_blur_enabled = False  # Track if face blur is enabled

# MediaPipe Pose Initialization
mp_pose = mp.solutions.pose.Pose()

# Video stream generator function


def generate_frames():
    global camera_index, face_blur_enabled
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Use the selected camera
    print(f'Cam index: {camera_index}')

    # Set resolution (example: 1280x720)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize YOLOv8 model for person detection
    model = YOLO("yolov8n.pt")
    # Use GPU if available
    if torch.cuda.is_available():
        model.to('cuda')
    else:
        print("Warning: CUDA not available. Using CPU instead.")
        model.to('cpu')

    while True:
        start_time = time.time()  # Start frame time
        success, frame = cap.read()
        if not success:
            print('Not successful')
            break

        # Flip the frame horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Run YOLOv8 object detection
        results = model(frame)
        boxes = results[0].boxes.cpu().numpy()  # Extract bounding boxes

        for box in boxes:
            if box.cls == 0:  # YOLO class 0 corresponds to 'person'
                x1, y1, x2, y2 = map(int, box.xyxy[0])

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

                    # Apply face blur only if the checkbox is enabled
                    if face_blur_enabled:
                        drawing.blur_face(person_frame, keypoints)

                # Place the processed person frame back into the original frame
                frame[y1:y2, x1:x2] = person_frame

        # Encode the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Calculate and print total frame processing time
        end_time = time.time()
        frame_time = end_time - start_time
        print(f"Total frame time: {frame_time:.2f} seconds")

        # Yield frame in byte format for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to set the camera


@app.route('/set_camera/<int:selected_camera_index>', methods=['GET'])
def set_camera(selected_camera_index):
    global camera_index
    camera_index = selected_camera_index  # Update the camera index based on user selection
    print(f'Selected: {camera_index}')
    return '', 200  # Return a 200 OK response to confirm

# Route to set face blur


@app.route('/set_face_blur/<string:state>', methods=['GET'])
def set_face_blur(state):
    global face_blur_enabled
    face_blur_enabled = (state == 'enabled')  # Enable or disable face blur
    print(f'Face blur {state}')
    return '', 200

# Route for the video feed


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main route


@app.route('/')
def index():
    return render_template('index.html', time=int(time.time()))


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
