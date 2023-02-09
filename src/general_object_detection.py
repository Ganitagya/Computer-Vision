# Homework Solution

# Importing the libraries
import cv2

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
cars_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')
buses_cascade = cv2.CascadeClassifier('haarcascade_buses_front.xml')
motorcycle_cascade = cv2.CascadeClassifier('haarcascade_motorcycle.xml')

video_src_camera = False  # is the source of the video a web cam
car_count = 0


# Defining a function that will do the detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.5, 60)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return frame


# Defining a function to detect cars, bikes, buses and trucks in a video
def detect_traffic(gray, frame):
    global car_count

    cars = cars_cascade.detectMultiScale(gray, 1.16, 1)
    # buses = buses_cascade.detectMultiScale(gray, 1.16, 1)
    # bikes = motorcycle_cascade.detectMultiScale(gray, 1.01, 1)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # cv2.putText(frame, "Car"+str(car_count), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        car_count += 1
    '''
    for (x, y, w, h) in buses:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
        
    for (x, y, w, h) in bikes:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
    '''

    return frame


if video_src_camera:

    # Doing some Face Recognition with the webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        _, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        cv2.imshow('Video', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

else:
    video_src = 'dataset/input_video.mp4'
    cap = cv2.VideoCapture(video_src)

    frame_cnt = 0

    while True:
        ret, frame = cap.read()

        frame_cnt += 1

        if frame_cnt % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect_traffic(gray, frame)
        '''
        cars = cars_cascade.detectMultiScale(gray, 1.1, 1)

        for (x,y,w,h) in cars:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) 
        '''

        cv2.imshow('video', canvas)

        if cv2.waitKey(33) == 27:
            break

    cv2.destroyAllWindows()
