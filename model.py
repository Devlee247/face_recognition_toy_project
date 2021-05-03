import cv2

# Face recognition has 4 layers.
# 1 : detection
# 2 : alignment
# 3 : representation
# 4 : verification
# -----------------------------

# 1 : detection by using opencv

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_COUNT, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ASCII 27 means 'esc'
while cv2.waitKey(27) < 0:
    ret, frame = capture.read()
    cv2.imshow("VideoFrame", frame)

capture.release()
cv2.destroyAllWindows()