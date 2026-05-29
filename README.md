FACE MASK DETECTION

A real-time face mask detection system built with Python and OpenCV using a webcam feed. This project detects human faces and determines whether a person is wearing a mask based on image-processing techniques such as color detection, grayscale variance analysis, and prediction smoothing.

FEATURES

1. Real-time webcam face detection

2. Mask / No Mask classification

3. Stable prediction using majority voting

4. Uses OpenCV DNN face 

5. Lightweight and fast

6. Works with standard webcams

TECHNOLOGIES USED
 
1. Python

2. OpenCV

3. NumPy
   
4. OpenCV DNN Face Detector

5. Python Collections Module


PROJECT STRUCTURE

project/

│── detect_mask_webcam.py

│── deploy.prototxt

│── res10_300x300_ssd_iter_140000.caffemodel

HOW TO RUN

1. Clone the repository:

git clone https://github.com/your-username/face-mask-detector.git

cd face-mask-detector

2. Make sure the following files are present:
   
detect_mask_webcam.py

deploy.prototxt

res10_300x300_ssd_iter_140000.caffemodel

3. Run the program:

python detect_mask_webcam.py 

4. Press Q to exit
