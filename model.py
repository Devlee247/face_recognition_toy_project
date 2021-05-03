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

face_cascade_name = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier()
if not face_cascade.load(cv2.samples.findFile(face_cascade_name)):
    print('Error loading face cascade')
    exit(0)

# ASCII 27 means 'esc'
while cv2.waitKey(27) < 0:
    ret, frame = capture.read()
    # error handling
    if not ret:
        print("Can't receive frame. Exiting ...")
        break
    # convert frame's color to gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray,
        minSize=(30,30)
        )
    print(faces)
    # draw rectangle
    for (x, y, w, h) in faces:
        dst = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0))

    cv2.imshow("VideoFrame", dst)

# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()