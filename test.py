from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime


video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load trained data
names_path = 'data/names.pkl'  
faces_path = 'data/face_data.pkl'

try:
    with open(names_path, 'rb') as f:
        LABELS = pickle.load(f)
    with open(faces_path, 'rb') as f:
        FACES = pickle.load(f)
    
    
    if len(FACES) != len(LABELS):
        print(f"Data mismatch: {len(FACES)} faces, {len(LABELS)} labels.")
        print("Please re-collect your face data to ensure both files match.")
        exit()
    
    # Train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3, weights='distance', metric='euclidean')
    knn.fit(FACES, LABELS)
    print(f"Model trained with {len(FACES)} face samples")
except FileNotFoundError:
    print("No training data found. Please run add_face.py first to collect face data.")
    exit()

if len(FACES) != len(LABELS):
    print(f"Data mismatch: {len(FACES)} faces, {len(LABELS)} labels.")
    print("Please re-collect your face data to ensure both files match.")
    exit()


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SMALL = cv2.FONT_HERSHEY_SIMPLEX

# Color scheme
COLORS = {
    'primary': (0, 150, 255),      # Orange
    'secondary': (255, 255, 255),   # White
    'success': (0, 255, 0),         # Green
    'warning': (0, 255, 255),       # Yellow
    'danger': (0, 0, 255),          # Red
    'info': (255, 255, 0),          # Cyan
    'dark': (50, 50, 50),           # Dark gray
    'light': (200, 200, 200)        # Light gray
}

def draw_info_panel(frame, recognized_faces, fps):
    """Draw information panel on the frame"""
    # Main info panel background
    cv2.rectangle(frame, (10, 10), (400, 150), COLORS['dark'], -1)
    cv2.rectangle(frame, (10, 10), (400, 150), COLORS['primary'], 2)
    
    # Title
    cv2.putText(frame, "FACIAL RECOGNITION SYSTEM", (20, 35), FONT, 0.7, COLORS['primary'], 2)
    
    # FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60), FONT_SMALL, 0.6, COLORS['secondary'], 1)
    
    # Time
    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"Time: {current_time}", (20, 85), FONT_SMALL, 0.6, COLORS['secondary'], 1)
    
    # Face count
    cv2.putText(frame, f"Faces Detected: {len(recognized_faces)}", (20, 110), FONT_SMALL, 0.6, COLORS['secondary'], 1)
    
    # Instructions
    cv2.putText(frame, "Press 'q' to quit", (20, 135), FONT_SMALL, 0.5, COLORS['info'], 1)

def draw_face_box(frame, x, y, w, h, name, confidence):
  # Draw enhanced face box with name and confidence

    if confidence > 0.7:
        color = COLORS['success']
        status = "HIGH"
    elif confidence > 0.5:
        color = COLORS['warning']
        status = "MEDIUM"
    else:
        color = COLORS['danger']
        status = "LOW"
    
    # Main face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    

    name_bg_height = 40
    cv2.rectangle(frame, (x, y - name_bg_height), (x + w, y), color, -1)
    cv2.rectangle(frame, (x, y - name_bg_height), (x + w, y), COLORS['secondary'], 1)
    
    # Name text
    cv2.putText(frame, name, (x + 5, y - 20), FONT_SMALL, 0.7, COLORS['secondary'], 2)
    

    conf_text = f"{confidence:.2f} ({status})"
    cv2.putText(frame, conf_text, (x + 5, y - 5), FONT_SMALL, 0.5, COLORS['secondary'], 1)
    
  
    corner_size = 10
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_size, y), color, 3)
    cv2.line(frame, (x, y), (x, y + corner_size), color, 3)
    # Top-right corner
    cv2.line(frame, (x + w - corner_size, y), (x + w, y), color, 3)
    cv2.line(frame, (x + w, y), (x + w, y + corner_size), color, 3)
    # Bottom-left corner
    cv2.line(frame, (x, y + h - corner_size), (x, y + h), color, 3)
    cv2.line(frame, (x, y + h), (x + corner_size, y + h), color, 3)
    # Bottom-right corner
    cv2.line(frame, (x + w - corner_size, y + h), (x + w, y + h), color, 3)
    cv2.line(frame, (x + w, y + h - corner_size), (x + w, y + h), color, 3)

def calculate_confidence(distances):
    # Calculate confidence based on distances to nearest neighbors
    if len(distances) == 0:
        return 0.0
    
    
    avg_distance = np.mean(distances)
    confidence = max(0, 1 - (avg_distance / 1000))  # Normalize to 0-1
    return confidence

# Main recognition loop
recognized_faces = []
fps_counter = 0
fps_start_time = time.time()
fps = 0  # Initialize fps variable

print("Starting Facial Recognition System...")
print("Press 'q' to quit")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize frame for better performance
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Enhanced face detection parameters
    faces = facedetect.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    recognized_faces = []
    
    for (x, y, w, h) in faces:
        # Extract and preprocess face with same preprocessing as training
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (50, 50))
        
        # Apply same preprocessing as in add_face.py
        face_roi = cv2.equalizeHist(face_roi)
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        face_roi = face_roi.flatten().reshape(1, -1)
        
        # Predict with confidence
        try:
            distances, indices = knn.kneighbors(face_roi)
            prediction = knn.predict(face_roi)[0]
            confidence = calculate_confidence(distances[0])
            
            recognized_faces.append({
                'name': prediction,
                'confidence': confidence,
                'bbox': (x, y, w, h)
            })
            
            # Draw enhanced face box
            draw_face_box(frame, x, y, w, h, prediction, confidence)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            # Draw unknown face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS['danger'], 2)
            cv2.putText(frame, "Unknown", (x, y - 10), FONT_SMALL, 0.7, COLORS['danger'], 2)
    
    # Calculate FPS
    fps_counter += 1
    if time.time() - fps_start_time >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_start_time = time.time()
    
    # Draw information panel
    draw_info_panel(frame, recognized_faces, fps)
    
    # Display frame
    cv2.imshow("Facial Recognition Security Camera", frame)
    
    # Handle key presses
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('s'):  # Save screenshot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Screenshot saved as {filename}")

# Cleanup
video.release()
cv2.destroyAllWindows()
print("Facial Recognition System stopped.")

