
from imutils.video import VideoStream, FPS
import numpy as np
import argparse
import imutils
import time
import cv2

DEFAULT_CONFIDENCE = 0.2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="Path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="Path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE, help="Minimum probability to filter weak predictions")
args = vars(ap.parse_args())


CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]
    print(f"[INFO] Frame dimensions: {frame.shape}")
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/127.5, size=(300, 300), mean=127.5, swapRB=True)
    net.setInput(blob)
    predictions = net.forward()
    for i in np.arange(0, predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        if confidence >= args["confidence"]:
            idx = int(predictions[0, 0, i, 1])
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            print(f"[INFO] Object detected: {label}")
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        print("[INFO] Exiting Program...")
        break

    fps.update()
fps.stop()
print(f"[INFO] Elapsed Time: {fps.elapsed():.2f} seconds")
print(f"[INFO] Approximate FPS: {fps.fps():.2f}")
cv2.destroyAllWindows()
vs.stop()
