import cv2
import numpy as np
kamera = cv2.VideoCapture(0)
detector = cv2.createBackgroundSubtractorMOG2(history= 100, varThreshold=40)
while True:
    _, frame = kamera.read()
    height, width, _ = frame.shape
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    min_ora = np.array([10, 50, 50])
    max_ora = np.array([30, 255, 255])
    min_ora2 = np.array([170, 50, 50])
    max_ora2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, min_ora, max_ora)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for crt in contours:
        area = cv2.contourArea(crt)

        if area > 1000:
            x, y, z, t = cv2.boundingRect(crt)
            cv2.rectangle(frame, (x, y), (x + z, y + t), (0, 255, 0), 3)

    cv2.imshow("pencere", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(30)

    if key == 27:
        break

kamera.release()
cv2.destroyAllWindows()

