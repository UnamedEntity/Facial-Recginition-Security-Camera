# Facial Recognition Security Camera System

A modern, high-accuracy facial recognition system with a beautiful UI and enhanced recognition capabilities.

## Features

### 🎯 **Enhanced Recognition Accuracy**
- Improved KNN classifier with distance-weighted voting
- Face quality assessment during data collection
- Advanced preprocessing with histogram equalization and noise reduction
- Confidence scoring for recognition results

### 🎨 **Beautiful Modern UI**
- Professional color scheme with status indicators
- Real-time FPS counter and system information
- Progress bars and visual feedback
- Corner indicators and enhanced face detection boxes
- Information panels with live statistics

### 📊 **Advanced Data Collection**
- Quality checks for lighting, contrast, and blur
- Progress tracking with visual feedback
- Configurable sample collection count
- Real-time quality assessment

### 🔧 **System Improvements**
- Error handling and graceful degradation
- Screenshot capture functionality (press 's')
- Better face detection parameters
- Optimized performance with frame resizing

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have the required files:**
   - `data/haarcascade_frontalface_default.xml` (OpenCV face detection model)
   - `data/` directory (will be created automatically)

## Usage

### 1. Collect Face Data
First, collect face data for recognition:

```bash
python add_face.py
```

**Features during collection:**
- Enter your name and desired sample count
- Real-time quality assessment
- Visual progress tracking
- Automatic data saving

**Tips for better accuracy:**
- Ensure good lighting conditions
- Move your head slightly for variety
- Keep your face clearly visible
- Avoid blurry or dark images

### 2. Run Recognition System
Start the facial recognition system:

```bash
python test.py
```

**Features during recognition:**
- Real-time face detection and recognition
- Confidence scoring with color-coded indicators
- Live system statistics
- Screenshot capture (press 's')

**Controls:**
- `q` - Quit the application
- `s` - Save a screenshot

## System Requirements

- **Python:** 3.7 or higher
- **Camera:** Webcam or USB camera
- **RAM:** Minimum 4GB recommended
- **Storage:** ~100MB for face data

## File Structure

```
Facial Recognition Security Camera/
├── add_face.py              # Face data collection script
├── test.py                  # Main recognition system
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── data/                   # Data directory
    ├── face_data.pkl       # Face feature data
    ├── names.pkl           # Name labels
    └── haarcascade_frontalface_default.xml  # Face detection model
```

## Technical Details

### Recognition Algorithm
- **Classifier:** K-Nearest Neighbors (KNN)
- **Parameters:** 3 neighbors, distance-weighted voting
- **Feature extraction:** 50x50 grayscale images
- **Preprocessing:** Histogram equalization, Gaussian blur

### Quality Assessment
- **Brightness check:** Ensures proper lighting
- **Contrast analysis:** Detects low-contrast images
- **Blur detection:** Uses Laplacian variance
- **Real-time feedback:** Color-coded status indicators

### Performance Optimizations
- Frame resizing for better performance
- Optimized face detection parameters
- Efficient data structures
- Memory management

## Troubleshooting

### Common Issues

1. **"No training data found"**
   - Run `add_face.py` first to collect face data

2. **Poor recognition accuracy**
   - Collect more face samples (100+ recommended)
   - Ensure good lighting during collection
   - Move your head for variety in training data

3. **Camera not detected**
   - Check camera connections
   - Ensure no other applications are using the camera

4. **Low FPS**
   - Reduce window size in the code
   - Close other applications
   - Check system resources

### Performance Tips

- Use good lighting for both collection and recognition
- Collect diverse face angles and expressions
- Regular system maintenance and updates
- Monitor system resources during operation

## Security Features

- Local data storage (no cloud dependencies)
- Confidence-based recognition
- Quality assessment prevents poor data
- Error handling for system stability

## Future Enhancements

- Multiple face detection support
- Database management interface
- Export/import functionality
- Advanced machine learning models
- Mobile app integration

## License

This project is open source and available under the MIT License.

## Support

For issues or questions, please check the troubleshooting section or create an issue in the repository.
