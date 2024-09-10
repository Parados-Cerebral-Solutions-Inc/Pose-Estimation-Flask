import cv2
import numpy as np
from math import sqrt, ceil
from typing import List, Tuple


import cv2
import numpy as np
import json
from math import sqrt, ceil
from typing import List, Tuple
from PIL import Image

# Define color variables
PARADOS_PRIMARY = (12, 137, 195)    # RGB: 0c89c3
PARADOS_SECONDARY = (222, 156, 24)  # RGB: de9c18

# Custom drawing config for FrontSquat
DRAWING_CONFIG = {
    "default": {
        'keypoints': {
            'focus': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
            'attributes': {
                'color': (255, 255, 255),  # Default white color for keypoints
                'radius': 20,
                'circle_thickness': 3,
                'outline_color': (0, 0, 0),  # Black outline
                'outline_thickness': 2,
                'fill_circle': False
            },
            'special_attributes': {
                'color': {
                    '11': PARADOS_SECONDARY,  # Left shoulder - orange
                    '12': PARADOS_PRIMARY,    # Right shoulder - blue
                    '13': PARADOS_SECONDARY,  # Left elbow - orange
                    '14': PARADOS_PRIMARY,    # Right elbow - blue
                    '15': PARADOS_SECONDARY,  # Left wrist - orange
                    '16': PARADOS_PRIMARY,    # Right wrist - blue
                    '23': PARADOS_SECONDARY,  # Left hip - orange
                    '24': PARADOS_PRIMARY,    # Right hip - blue
                    '25': PARADOS_SECONDARY,  # Left knee - orange
                    '26': PARADOS_PRIMARY,    # Right knee - blue
                    '27': PARADOS_SECONDARY,  # Left ankle - orange
                    '28': PARADOS_PRIMARY     # Right ankle - blue
                },
            },
        },
        'connections': {
            'focus': [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
            'attributes': {
                'color': (255, 255, 255),  # Black color for connections
                'line_thickness': 3,
                'outline_color': (0, 0, 0),  # White outline
                'outline_thickness': 2,
                'spacing': 20
            },
            'special_attributes': {}
        }
    }
}

"""
Parados All Rights Reserved *** Add year of company establishment, etc.. here***
Filename: paradosDrawingUtils.py
Original Author: Benjamin MacPherson
Contributing Author: exampleContributor1, exampleContributor2
Description: This module provides custom drawing utilities for visualizing pose estimation landmarks and connections using the MediaPipe framework. It includes functionality for dynamically configuring drawing settings based on visibility confidence, outline, and other aesthetic preferences.

This script allows for the customization of keypoints and connections drawing, including dynamic color based on confidence levels. It supports flexible configurations through a centralized configuration file, making it easy to adapt to different visualization requirements.

Dependencies:
- OpenCV: Used for drawing on image frames.
- NumPy: Provides numerical operations for vector calculations.
- MediaPipe: Provides the pose estimation technology.
- paradosConfig: Contains configuration settings for drawing preferences.

Classes:
- CustomDrawingSpec: A class to encapsulate the drawing specifications for keypoints and connections.

Functions:
- confidence_to_color(confidence): Converts confidence to a color ranging from red (low) to green (high).
- draw_custom_landmarks(image, landmark_list, connections, landmark_drawing_spec, connection_drawing_spec, draw_landmarks_config): Draws the landmarks and the connections on the image.
- _normalized_to_pixel_coordinates(normalized_x, normalized_y, image_width, image_height): Converts normalized coordinates to pixel coordinates.
- draw(image, landmark_list, connections): Main function to handle the drawing process using the configuration settings.
- convert_line_to_keypoint_pair(csv_line): Converts a keypoints CSV output line to an (x, y) pair list for use in the drawing function.
- draw_keypoints_and_connections(image, keypoints, movement_type, override_json): Draws keypoints and connections on the image with optional movement type and override configurations.

Usage:
The functions are typically called within a video processing loop to draw pose estimation results on each frame, allowing for real-time visualization of the pose estimation process.

Updates and Versions:
Version 0.1: Initial implementation of the custom drawing utilities for pose estimation - Benjamin MacPherson
Version 0.1.5: Temporary transitionary point between using old drawing functionality including the mediapipe built-in drawing functionality and the new draw utils that uses draw config overrides to modify the way the drawing output is applied - Benjamin MacPherson
Version 0.2: Changed the Drawing Overrider structure to support color change for all keypoints and connections, radius, thickness, outline color, and outline thickness for keypoints, and line thickness for connections - Yehor Sanko
Version 0.3: Added the ability to blur the face based on detected facial keypoints - Yehor Sanko
"""


class CustomDrawingSpec:
    """
    A class to encapsulate the drawing specifications for keypoints and connections.

    Attributes:
    - color: Color for drawing the annotation. Default to white.
    - thickness: Thickness for drawing the annotation. Default to 2 pixels.
    - circle_radius: Circle radius for keypoints. Default to 2 pixels.
    - fill_circle: Whether to fill the keypoint circles. Default to True.
    - outline_color: Color for the outline of the keypoints. Default to black.
    - outline_thickness: Thickness of the keypoint outline. Default to 1 pixel.
    """

    def __init__(self,
                 color=(224, 224, 224),
                 thickness=2,
                 circle_radius=2,
                 fill_circle=True,
                 outline_color=(0, 0, 0),
                 outline_thickness=1):
        self.color = color
        self.thickness = thickness
        self.circle_radius = circle_radius
        self.fill_circle = fill_circle
        self.outline_color = outline_color
        self.outline_thickness = outline_thickness


KEYPOINT_DICT = {
    0: ['nose', 1, 4],
    1: ['left_eye_inner', 0, 2],
    2: ['left_eye', 1, 3],
    3: ['left_eye_outer', 2, 7],
    4: ['right_eye_inner', 0, 5],
    5: ['right_eye', 4, 6],
    6: ['right_eye_outer', 5, 8],
    7: ['left_ear', 3],
    8: ['right_ear', 6],
    9: ['mouth_left', 10],
    10: ['mouth_right', 9],
    11: ['left_shoulder', 12, 13, 23],
    12: ['right_shoulder', 11, 14, 24],
    13: ['left_elbow', 11, 15],
    14: ['right_elbow', 12, 16],
    15: ['left_wrist', 13, 17, 21],
    16: ['right_wrist', 14, 18, 22],
    17: ['left_pinky', 15, 19],
    18: ['right_pinky', 16, 20],
    19: ['left_index', 15, 17],
    20: ['right_index', 16, 18],
    21: ['left_thumb', 15],
    22: ['right_thumb', 16],
    23: ['left_hip', 11, 24, 25],
    24: ['right_hip', 12, 23, 26],
    25: ['left_knee', 23, 27],
    26: ['right_knee', 24, 28],
    27: ['left_ankle', 25, 29, 31],
    28: ['right_ankle', 26, 30, 32],
    29: ['left_heel', 27, 31],
    30: ['right_heel', 28, 32],
    31: ['left_foot', 27, 29],
    32: ['right_foot', 28, 30],
}


def convert_line_to_keypoint_pair(csv_line: str) -> List[Tuple[float, float]]:
    """
    Converts a keypoints CSV output line to an (x, y) pair list for use in the drawing function.

    Parameters:
    - csv_line (str): A line from a CSV file containing keypoints.

    Returns:
    - List[Tuple[float, float]]: A list of (x, y) coordinate pairs.
    """
    values = csv_line.split(',')
    keypoints = [(float(values[i]), float(values[i + 1])) for i in range(1, len(values), 4)]
    return keypoints


def calculate_scaling_factor(image_width: int, image_height: int, baseline: int = 1080) -> float:
    """
    Calculates the scaling factor based on the smaller dimension of the image and a baseline dimension.

    Parameters:
    - image_width (int): The width of the image.
    - image_height (int): The height of the image.
    - baseline (int): The baseline dimension for scaling. Default is 1080.

    Returns:
    - float: The scaling factor.
    """
    smaller_dimension = min(image_width, image_height)
    return smaller_dimension / baseline


def draw_keypoints_and_connections(image: np.ndarray, keypoints: List[Tuple[float, float]]):
    """
    Draws keypoints and connections on the image with optional movement type and override configurations.

    Parameters:
    - image (np.ndarray): The image on which to draw.
    - keypoints (List[Tuple[float, float]]): List of (x, y) coordinates in normalized values.
    """
    # Load default drawing configuration
    config = DRAWING_CONFIG.get("default", {})

    keypoints_config = config['keypoints']
    connections_config = config['connections']

    keypoints_focus = keypoints_config.get('focus', [])
    keypoints_attributes = keypoints_config.get('attributes', {})
    keypoints_special_attributes = keypoints_config.get('special_attributes', {})

    connections_focus = connections_config.get('focus', [])
    connections_attributes = connections_config.get('attributes', {})
    connections_special_attributes = connections_config.get('special_attributes', {})

    image_height, image_width, _ = image.shape

    # Calculate the scaling factor based on the image dimensions
    scaling_factor = calculate_scaling_factor(image_width, image_height)

    def apply_keypoint_attributes(idx, default_attrs, special_attrs):
        # Start with default attributes for keypoints
        color = tuple(default_attrs.get('color', (255, 255, 255)))
        radius = max(1, ceil(default_attrs.get('radius', 5) * sqrt(scaling_factor)))  # Minimum radius of 1
        circle_thickness = max(1, ceil(default_attrs.get('circle_thickness', 2) * scaling_factor))  # Minimum thickness of 1
        outline_color = tuple(default_attrs.get('outline_color', (0, 0, 0)))
        outline_thickness = max(0, ceil(default_attrs.get('outline_thickness', 1) * scaling_factor))  # Minimum thickness of 0
        fill_circle = default_attrs.get('fill_circle', False)

        # Apply special attributes if they exist
        for _, special_attr_dict in special_attrs.items():
            if str(idx) in special_attr_dict:
                color = tuple(special_attrs.get('color', {}).get(str(idx), color))
                radius = max(1, ceil(special_attrs.get('radius', {}).get(str(idx), radius) * sqrt(scaling_factor)))
                circle_thickness = max(1, ceil(special_attrs.get('circle_thickness', {}).get(str(idx), circle_thickness) * scaling_factor))
                outline_color = tuple(special_attrs.get('outline_color', {}).get(str(idx), outline_color))
                outline_thickness = max(0, ceil(special_attrs.get('outline_thickness', {}).get(str(idx), outline_thickness) * scaling_factor))
                fill_circle = special_attrs.get('fill_circle', {}).get(str(idx), fill_circle)

        return color, radius, circle_thickness, outline_color, outline_thickness, fill_circle

    def apply_connection_attributes(idx, default_attrs, special_attrs):
        # Start with default attributes for connections
        color = tuple(default_attrs.get('color', (255, 255, 255)))
        line_thickness = max(1, ceil(default_attrs.get('line_thickness', 2) * scaling_factor))  # Minimum thickness of 1
        outline_color = tuple(default_attrs.get('outline_color', (0, 0, 0)))
        outline_thickness = max(0, ceil(default_attrs.get('outline_thickness', 1) * scaling_factor))  # Minimum thickness of 0
        spacing = max(0, ceil(default_attrs.get('spacing', 0) * scaling_factor))  # Minimum spacing of 0

        # Apply special attributes if they exist
        for _, special_attr_dict in special_attrs.items():
            if str(idx) in special_attr_dict:
                color = tuple(special_attrs.get('color', {}).get(str(idx), color))
                line_thickness = max(1, ceil(special_attrs.get('line_thickness', {}).get(str(idx), line_thickness) * scaling_factor))
                outline_color = tuple(special_attrs.get('outline_color', {}).get(str(idx), outline_color))
                outline_thickness = max(0, ceil(special_attrs.get('outline_thickness', {}).get(str(idx), outline_thickness) * scaling_factor))
                spacing = max(0, ceil(special_attrs.get('spacing', {}).get(str(idx), spacing) * scaling_factor))

        return color, line_thickness, outline_color, outline_thickness, spacing

    # Draw the connections between keypoints
    for idx, (x, y) in enumerate(keypoints):

        # Check for NaN
        if np.isnan(x) or np.isnan(y):
            continue

        if idx not in connections_focus:
            continue

        for connected_idx in KEYPOINT_DICT[idx][1:]:
            if connected_idx < len(keypoints) and connected_idx in connections_focus:
                x2, y2 = keypoints[connected_idx]

                # Check for NaN
                if np.isnan(x2) or np.isnan(y2):
                    continue

                # Get attributes for connections
                color, line_thickness, outline_color, outline_thickness, spacing = apply_connection_attributes(
                    idx, connections_attributes, connections_special_attributes)

                # Get radii for start and end keypoints
                _, radius_start, *_ = apply_keypoint_attributes(
                    idx, keypoints_attributes, keypoints_special_attributes)
                _, radius_end, *_ = apply_keypoint_attributes(
                    connected_idx, keypoints_attributes, keypoints_special_attributes)

                x_px = int(x * image_width)
                y_px = int(y * image_height)
                x2_px = int(x2 * image_width)
                y2_px = int(y2 * image_height)

                # Calculate the direction vector
                direction = np.array([x2_px, y2_px]) - np.array([x_px, y_px])
                distance = np.linalg.norm(direction)

                # Ensure the direction vector is not zero
                if distance == 0:
                    continue

                # Calculate the minimum distance required to draw the connection
                min_distance = radius_start + radius_end + 2 * spacing + 2

                if distance < min_distance:
                    continue

                # Normalize the direction vector
                direction = direction / distance

                # Adjust start and end points based on radii and spacing
                start_point_offset = (int(x_px + direction[0] * (radius_start + spacing)),
                                      int(y_px + direction[1] * (radius_start + spacing)))
                end_point_offset = (int(x2_px - direction[0] * (radius_end + spacing)),
                                    int(y2_px - direction[1] * (radius_end + spacing)))

                if np.isnan(start_point_offset).any() or np.isnan(end_point_offset).any():
                    continue

                # Draw the outline first
                if outline_thickness > 0:
                    cv2.line(image, start_point_offset, end_point_offset, outline_color, line_thickness + (outline_thickness * 2), lineType=cv2.LINE_AA)

                # Draw the main line on top
                cv2.line(image, start_point_offset, end_point_offset, color, line_thickness, lineType=cv2.LINE_AA)

    # Draw the circles for each keypoint in focus
    for idx, (x, y) in enumerate(keypoints):
        if idx not in connections_focus:
            continue

        # Check for NaN
        if np.isnan(x) or np.isnan(y):
            continue

        # Get attributes for keypoints
        color, radius, circle_thickness, outline_color, outline_thickness, fill_circle = apply_keypoint_attributes(
            idx, keypoints_attributes, keypoints_special_attributes)

        x_px = int(x * image_width)
        y_px = int(y * image_height)

        # Draw the outline first
        if outline_thickness > 0:
            cv2.circle(image, (x_px, y_px), radius, outline_color, circle_thickness + (outline_thickness * 2), lineType=cv2.LINE_AA)

        # Draw the filled or hollow circle on top
        if fill_circle:
            cv2.circle(image, (x_px, y_px), radius, color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(image, (x_px, y_px), radius, color, circle_thickness, lineType=cv2.LINE_AA)


# Face blur configuration
FACE_BLUR_CONFIG = {
    "BLUR_FACE_EXTEND": {
        "top": 1.8,
        "bottom": 1.0,
        "left": 0.5,
        "right": 0.5,
    },
    "PIXELATION_PERCENTAGE": 15,  # 1% to 100%
    "BLUR_FACE_STRENGTH": 0.75  # 0 to 1
}


def blur_face(image: np.ndarray, keypoints: List[Tuple[float, float]]):
    """
    Blurs the area around the face based on detected facial keypoints.

    Parameters:
    - image (np.ndarray): The image on which to draw.
    - keypoints (List[Tuple[float, float]]): List of (x, y) coordinates in normalized values.
    """

    facial_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    facial_points = []

    for i in facial_indices:
        if i < len(keypoints):
            try:
                x = keypoints[i][0]
                y = keypoints[i][1]
                if not np.isnan(x) and not np.isnan(y):
                    facial_points.append((x, y))
            except (ValueError, TypeError):
                print(f"Invalid keypoint data at index {i}: {keypoints[i]}")

    if not facial_points:
        return

    image_height, image_width, _ = image.shape
    x_coords = [int(point[0] * image_width) for point in facial_points]
    y_coords = [int(point[1] * image_height) for point in facial_points]

    if not x_coords or not y_coords:
        return

    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    # Apply the extension proportions from FACE_BLUR_CONFIG
    top_extend = int(FACE_BLUR_CONFIG["BLUR_FACE_EXTEND"]["top"] * (y_max - y_min))
    bottom_extend = int(FACE_BLUR_CONFIG["BLUR_FACE_EXTEND"]["bottom"] * (y_max - y_min))
    left_extend = int(FACE_BLUR_CONFIG["BLUR_FACE_EXTEND"]["left"] * (x_max - x_min))
    right_extend = int(FACE_BLUR_CONFIG["BLUR_FACE_EXTEND"]["right"] * (x_max - x_min))

    x_min = max(0, x_min - left_extend)
    x_max = min(image_width, x_max + right_extend)
    y_min = max(0, y_min - top_extend)
    y_max = min(image_height, y_max + bottom_extend)

    face_area = image[y_min:y_max, x_min:x_max]

    # Apply pixelation
    pixelation_percentage = max(1, min(FACE_BLUR_CONFIG["PIXELATION_PERCENTAGE"], 100))
    pixelation_size = max(1, 100 // pixelation_percentage)

    if face_area.shape[0] // pixelation_size == 0 or face_area.shape[1] // pixelation_size == 0:
        return

    small = cv2.resize(face_area, (face_area.shape[1] // pixelation_size, face_area.shape[0] // pixelation_size), interpolation=cv2.INTER_LINEAR)
    pixelated_face_cv = cv2.resize(small, (face_area.shape[1], face_area.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply Gaussian blur on top of the pixelation
    blur_strength = max(1, int(FACE_BLUR_CONFIG["BLUR_FACE_STRENGTH"] * 100))
    blurred_face = cv2.GaussianBlur(pixelated_face_cv, (blur_strength | 1, blur_strength | 1), 0)

    # Apply pixelation using PIL
    pil_image = Image.fromarray(blurred_face)
    small_pil = pil_image.resize((blurred_face.shape[1] // pixelation_size, blurred_face.shape[0] // pixelation_size), resample=Image.BICUBIC)
    pixelated_pil = small_pil.resize(pil_image.size, Image.NEAREST)
    pixelated_face_pil = np.array(pixelated_pil)

    image[y_min:y_max, x_min:x_max] = pixelated_face_pil
