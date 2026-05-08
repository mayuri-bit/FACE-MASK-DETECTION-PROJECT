import cv2
import numpy as np
from collections import deque

# Load face detector
prototxt = "face_detector/deploy.prototxt"
weights = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNet(prototxt, weights)

# Start webcam
cap = cv2.VideoCapture(0)

# Store last few predictions (for smoothing)
history = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    label = "Detecting..."
    color = (255, 255, 0)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(w, endX), min(h, endY)

            face = frame[startY:endY, startX:endX]

            if face.shape[0] == 0 or face.shape[1] == 0:
                continue

            # Split face
            mid = face.shape[0] // 2
            upper = face[:mid, :]
            lower = face[mid:, :]

            upper_gray = cv2.cvtColor(upper, cv2.COLOR_BGR2GRAY)
            lower_gray = cv2.cvtColor(lower, cv2.COLOR_BGR2GRAY)

            upper_var = np.var(upper_gray)
            lower_var = np.var(lower_gray)

            upper_mean = np.mean(upper_gray)
            lower_mean = np.mean(lower_gray)

            lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)

            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])

            mask_blue = cv2.inRange(lower_hsv, lower_blue, upper_blue)
            blue_ratio = np.sum(mask_blue) / (mask_blue.size)

            # Raw prediction
            if blue_ratio > 0.1 or (lower_var < upper_var * 0.8 and lower_mean < upper_mean):
                pred = "Mask"
            else:
                pred = "No Mask"

            # Add to history
            history.append(pred)

            # Majority vote (stabilization)
            if history.count("Mask") > history.count("No Mask"):
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), color, 2)

            break  # only process one face for stability

    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    cv2.imshow("Stable Mask Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()