import cv2
import os

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
while True:
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
        minSize=(100,100)
        )
    # print(faces)

    # draw rectangle
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0))
    
    if len(faces) != 0:
        roi = frame[ faces[0][1]:faces[0][1]+faces[0][3] , faces[0][0]:faces[0][0]+faces[0][2] ]

    cv2.imshow("VideoFrame", frame)

    # when you press 'c', capture your face
    # when you press 'Esc', turn off camera
    if cv2.waitKey(1) == ord('c') and len(faces) != 0:
        print("Captured!")
        
        # file count in face_data directory
        file_name = './face_data/' + str(len(os.listdir('./face_data'))) + '.png'

        # image write in face_data directory
        cv2.imwrite(file_name, roi)

    elif cv2.waitKey(1) == 27:
        print("Turn of Camera!")
        break

    


# When everything done, release the capture
capture.release()
cv2.destroyAllWindows()