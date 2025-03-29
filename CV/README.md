# Drowsiness Detection System

## Overview
This project is a real-time **Drowsiness Detection System** that monitors a person's eye activity using a webcam and raises an alarm if signs of drowsiness are detected. The system utilizes **OpenCV**, **dlib**, and **scipy.spatial.distance** to analyze eye aspect ratio (EAR) and detect if the user’s eyes are closing over time.

## Features
- **Real-time face and eye detection** using `dlib` and `OpenCV`
- **Eye Aspect Ratio (EAR) computation** to determine drowsiness
- **Alarm System** that triggers an audio alert when drowsiness is detected
- **Efficient performance** with frame optimization for real-time detection

## Technologies Used
- **Python** (Core language)
- **OpenCV** (`cv2` for real-time video capture & processing)
- **dlib** (Pre-trained face and eye detection model)
- **scipy.spatial.distance** (Calculates the EAR)
- **imutils** (Convenient image processing functions)
- **pygame.mixer** (Plays alarm sounds)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/drowsiness-detection.git
   cd drowsiness-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the `dlib` facial landmark predictor model:
   ```bash
   wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
   ```

## How to Run
Run the main script using:
```bash
python drowsiness_detection.py
```

## How It Works
1. Captures real-time video from the webcam
2. Detects facial landmarks using `dlib`
3. Computes the Eye Aspect Ratio (EAR) to determine eye closure
4. If EAR is below a threshold for a sustained period, an alarm sound is played

## Configuration
- **Threshold EAR:** Default is `0.25` (Modify as needed)
- **Alarm Sound:** Replace `music.wav` with your preferred alert sound
- **Camera Index:** Modify `cv2.VideoCapture(0)` if using an external webcam

## Future Enhancements
- Add a **GUI** for user interaction
- Improve **blink detection** for better accuracy
- Optimize model for **lower computational cost**

## License
This project is open-source under the **MIT License**.

## Acknowledgments
- `dlib` for providing robust facial landmark detection models
- `OpenCV` for computer vision capabilities

## Contact
For issues or improvements, feel free to contribute or raise an issue on GitHub.
