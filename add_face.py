import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# UI Constants
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

def ensure_data_directory():
    """Ensure the data directory exists"""
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created data directory")

def draw_collection_ui(frame, face_data, target_count, current_face=None):
    """Draw the face collection UI"""
    # Main info panel
    cv2.rectangle(frame, (10, 10), (500, 200), COLORS['dark'], -1)
    cv2.rectangle(frame, (10, 10), (500, 200), COLORS['primary'], 2)
    
    # Title
    cv2.putText(frame, "FACE DATA COLLECTION", (20, 35), FONT, 0.8, COLORS['primary'], 2)
    
    # Progress information
    progress = len(face_data) / target_count
    cv2.putText(frame, f"Progress: {len(face_data)}/{target_count} ({progress*100:.1f}%)", 
                (20, 65), FONT_SMALL, 0.6, COLORS['secondary'], 1)
    
    # Progress bar
    bar_width = 460
    bar_height = 20
    bar_x, bar_y = 20, 85
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLORS['light'], -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), COLORS['primary'], 2)
    
    filled_width = int(bar_width * progress)
    if filled_width > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), COLORS['success'], -1)
    
    # Instructions
    cv2.putText(frame, "Instructions:", (20, 125), FONT_SMALL, 0.6, COLORS['info'], 1)
    cv2.putText(frame, "1. Position your face in the camera", (20, 145), FONT_SMALL, 0.5, COLORS['secondary'], 1)
    cv2.putText(frame, "2. Move your head slightly for variety", (20, 165), FONT_SMALL, 0.5, COLORS['secondary'], 1)
    cv2.putText(frame, "3. Press 'q' to quit early", (20, 185), FONT_SMALL, 0.5, COLORS['secondary'], 1)
    
    # Face detection status
    if current_face is not None:
        cv2.putText(frame, "Face Detected - Collecting Data", (WINDOW_WIDTH - 350, 30), 
                    FONT_SMALL, 0.7, COLORS['success'], 2)
    else:
        cv2.putText(frame, "No Face Detected", (WINDOW_WIDTH - 250, 30), 
                    FONT_SMALL, 0.7, COLORS['warning'], 2)

def draw_face_guide(frame, x, y, w, h):
    """Draw face guide overlay"""
    # Main face rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLORS['success'], 2)
    
    # Guide text
    cv2.putText(frame, "FACE DETECTED", (x, y - 10), FONT_SMALL, 0.7, COLORS['success'], 2)
    
    # Corner indicators
    corner_size = 15
    # Top-left corner
    cv2.line(frame, (x, y), (x + corner_size, y), COLORS['success'], 3)
    cv2.line(frame, (x, y), (x, y + corner_size), COLORS['success'], 3)
    # Top-right corner
    cv2.line(frame, (x + w - corner_size, y), (x + w, y), COLORS['success'], 3)
    cv2.line(frame, (x + w, y), (x + w, y + corner_size), COLORS['success'], 3)
    # Bottom-left corner
    cv2.line(frame, (x, y + h - corner_size), (x, y + h), COLORS['success'], 3)
    cv2.line(frame, (x, y + h), (x + corner_size, y + h), COLORS['success'], 3)
    # Bottom-right corner
    cv2.line(frame, (x + w - corner_size, y + h), (x + w, y + h), COLORS['success'], 3)
    cv2.line(frame, (x + w, y + h - corner_size), (x + w, y + h), COLORS['success'], 3)

def preprocess_face(face_img):
    """Enhanced face preprocessing for better recognition"""
    # Convert to grayscale if not already
    if len(face_img.shape) == 3:
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Resize to standard size
    face_img = cv2.resize(face_img, (50, 50))
    
    # Apply histogram equalization for better contrast
    face_img = cv2.equalizeHist(face_img)
    
    # Apply Gaussian blur to reduce noise
    face_img = cv2.GaussianBlur(face_img, (3, 3), 0)
    
    return face_img

def check_face_quality(face_img):
    """Check if the face image is of good quality"""
    # Check brightness
    mean_brightness = np.mean(face_img)
    if mean_brightness < 30 or mean_brightness > 225:
        return False, "Poor lighting"
    
    # Check contrast
    std_dev = np.std(face_img)
    if std_dev < 20:
        return False, "Low contrast"
    
    # Check if face is too blurry
    laplacian_var = cv2.Laplacian(face_img, cv2.CV_64F).var()
    if laplacian_var < 100:
        return False, "Too blurry"
    
    return True, "Good quality"

# Main collection process
def collect_face_data():
    """Main function to collect face data"""
    ensure_data_directory()
    
    print("=== FACE DATA COLLECTION ===")
    name = input("Enter your name: ").strip()
    if not name:
        print("Name cannot be empty!")
        return
    
    # Get target count
    target_count = 100
    try:
        user_count = input(f"Enter number of samples to collect (default: {target_count}): ").strip()
        if user_count:
            target_count = int(user_count)
    except ValueError:
        print(f"Invalid input, using default: {target_count}")
    
    face_data = []
    i = 0
    collection_start_time = time.time()
    
    print(f"\nStarting collection for '{name}'...")
    print("Position your face in the camera and move slightly for variety.")
    print("Press 'q' to quit early.\n")
    
    while len(face_data) < target_count:
        ret, frame = video.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize frame for better performance
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced face detection
        faces = facedetect.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        current_face = None
        
        for (x, y, w, h) in faces:
            current_face = (x, y, w, h)
            
            # Extract face region
            face_roi = frame[y:y + h, x:x + w]
            
            # Preprocess face
            processed_face = preprocess_face(face_roi)
            
            # Check quality
            is_good_quality, quality_msg = check_face_quality(processed_face)
            
            # Collect data every 10 frames if quality is good
            if len(face_data) < target_count and i % 10 == 0 and is_good_quality:
                face_data.append(processed_face)
                print(f"Collected sample {len(face_data)}/{target_count} - {quality_msg}")
            elif not is_good_quality and i % 30 == 0:
                print(f"Quality check: {quality_msg}")
            
            # Draw face guide
            draw_face_guide(frame, x, y, w, h)
            
            # Show quality status
            color = COLORS['success'] if is_good_quality else COLORS['warning']
            cv2.putText(frame, quality_msg, (x, y + h + 25), FONT_SMALL, 0.6, color, 1)
        
        i += 1
        
        # Draw UI
        draw_collection_ui(frame, face_data, target_count, current_face)
        
        # Display frame
        cv2.imshow("Face Data Collection", frame)
        
        # Handle key presses
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
    
    # Calculate collection time
    collection_time = time.time() - collection_start_time
    
    # Cleanup
    video.release()
    cv2.destroyAllWindows()
    
    if len(face_data) == 0:
        print("No face data collected!")
        return
    
    # Process collected data
    print(f"\nCollection completed!")
    print(f"Samples collected: {len(face_data)}")
    print(f"Time taken: {collection_time:.1f} seconds")
    print(f"Average time per sample: {collection_time/len(face_data):.2f} seconds")
    
    # Convert to numpy array
    face_data = np.asarray(face_data)
    face_data = face_data.reshape(len(face_data), -1)
    
    # Save data
    save_face_data(name, face_data)
    
    print(f"\nFace data for '{name}' saved successfully!")

def save_face_data(name, face_data):
    """Save face data to pickle files"""
    # Load or initialize names and faces
    names_path = 'data/names.pkl'
    faces_path = 'data/face_data.pkl'
    
    if os.path.exists(names_path):
        with open(names_path, 'rb') as f:
            names = pickle.load(f)
    else:
        names = []
    
    if os.path.exists(faces_path):
        with open(faces_path, 'rb') as f:
            faces = pickle.load(f)
    else:
        faces = np.empty((0, face_data.shape[1]), dtype=face_data.dtype)
    
    # Append new data
    names.extend([name] * len(face_data))
    faces = np.append(faces, face_data, axis=0)
    
    # Ensure synchronization
    if len(names) != len(faces):
        print("Error: Mismatch between names and faces. Aborting save.")
        return
    
    # Save updated data
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)
    
    print(f"Total faces in database: {len(faces)}")
    print(f"Total names in database: {len(names)}")

if __name__ == "__main__":
    try:
        collect_face_data()
    except KeyboardInterrupt:
        print("\nCollection interrupted by user.")
    except Exception as e:
        print(f"Error during collection: {e}")
    finally:
        if video.isOpened():
            video.release()
        cv2.destroyAllWindows()

    