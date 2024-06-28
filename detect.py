import cv2
import numpy as np
import os
import urllib.request
import collections
from gesture_recognition import GestureRecognizer

# Create directory to save YOLO files
if not os.path.exists("yolo-coco"):
    os.makedirs("yolo-coco")

# URLs of the YOLO files (use YOLOv3-tiny for faster detection)
yolo_cfg_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
yolo_weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
coco_names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Paths to save the YOLO files
yolo_cfg_path = "yolo-coco/yolov3-tiny.cfg"
yolo_weights_path = "yolo-coco/yolov3-tiny.weights"
coco_names_path = "yolo-coco/coco.names"

# Download YOLO cfg
if not os.path.exists(yolo_cfg_path):
    print("Downloading yolov3-tiny.cfg...")
    urllib.request.urlretrieve(yolo_cfg_url, yolo_cfg_path)

# Download YOLO weights
if not os.path.exists(yolo_weights_path):
    print("Downloading yolov3-tiny.weights...")
    urllib.request.urlretrieve(yolo_weights_url, yolo_weights_path)

# Download COCO names
if not os.path.exists(coco_names_path):
    print("Downloading coco.names...")
    urllib.request.urlretrieve(coco_names_url, coco_names_path)

# Load YOLO
net = cv2.dnn.readNet(yolo_weights_path, yolo_cfg_path)

# Uncomment these lines if you have a CUDA-capable GPU
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load class names
with open(coco_names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Reduce resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Data structures to store paths and heatmap
object_paths = collections.defaultdict(list)
object_speeds = collections.defaultdict(list)

# Assuming the initial heatmap size as a typical 720p frame size
# This will be resized dynamically according to the frame dimensions.
heatmap = None

frame_count = 0
frame_skip = 10  # Process every 10th frame to reduce computational load
speed_threshold = 20  # Speed threshold for anomaly detection
movement_threshold = 5  # Minimum movement threshold to ignore small movements
history_length = 5  # Number of frames to consider for speed averaging

gesture_recognizer = GestureRecognizer()

def calculate_speed(point1, point2, fps):
    """Calculate speed between two points."""
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    return distance * fps

def is_fast_movement(speeds, threshold):
    """Determine if the movement is fast based on speed history."""
    if len(speeds) < history_length:
        return False
    recent_speeds = speeds[-history_length:]
    return np.mean(recent_speeds) > threshold

def detect_and_track_with_visualization():
    global frame_count, heatmap
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        height, width, _ = frame.shape
        
        # Initialize heatmap with the frame dimensions if not already done
        if heatmap is None:
            heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Object detection processing
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.6:  # Increase the confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                # Tracking paths
                object_id = f"{label}_{i}"
                center_point = (x + w // 2, y + h // 2)
                object_paths[object_id].append(center_point)
                gesture_recognizer.update_movements(object_id, center_point)
                
                if len(object_paths[object_id]) > 1:
                    speed = calculate_speed(object_paths[object_id][-2], object_paths[object_id][-1], fps)
                    if speed > movement_threshold:  # Ignore very small movements
                        object_speeds[object_id].append(speed)
                    
                    if is_fast_movement(object_speeds[object_id], speed_threshold):
                        cv2.putText(frame, "FAST", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # Update heatmap
                cv2.circle(heatmap, center_point, 5, 1, thickness=-1)
                
                color = (0, 255, 0)  # Green color for bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Recognize gestures
        gesture_recognizer.recognize_gestures()
        
        # Visualize paths
        for object_id, path in object_paths.items():
            if len(path) > 1:
                for j in range(len(path) - 1):
                    cv2.line(frame, path[j], path[j + 1], (0, 0, 255), 2)
                if len(path) > history_length:
                    object_paths[object_id] = object_paths[object_id][-history_length:]
        
        # Display the resulting frame
        cv2.imshow("Object Detection and Path Tracking", frame)
        
        # Create heatmap visualization
        heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        
        # Resize heatmap_colored to match frame size
        heatmap_colored_resized = cv2.resize(heatmap_colored, (width, height))
        
        # Overlay the heatmap on the frame
        heatmap_overlay = cv2.addWeighted(frame, 0.7, heatmap_colored_resized, 0.3, 0)
        cv2.imshow("Heatmap", heatmap_overlay)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Main script
if __name__ == "__main__":
    detect_and_track_with_visualization()
        