import cv2
import numpy as np
import imutils
import datetime

# Load the Haar cascade classifier
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Check if the cascade was loaded successfully
if gun_cascade.empty():
    print("Error: Could not load gun cascade XML file.")
    exit()

# Start video capture from the default camera
camera = cv2.VideoCapture(2)

if not camera.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# Flag to check if a gun was detected
gun_exists = False

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for consistency
    frame = imutils.resize(frame, width=500)

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    guns = gun_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(100, 100))

    # If guns are detected, draw rectangles and mark detection
    if len(guns) > 0:
        gun_exists = True
        for (x, y, w, h) in guns:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, "Gun Detected!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Add timestamp to frame
    timestamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, timestamp, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Security Feed", frame)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

# Final detection result
if gun_exists:
    print("Gun detected!")
else:
    print("No gun detected.")
