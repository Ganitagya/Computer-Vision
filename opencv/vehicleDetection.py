import cv2
import numpy as np

# capture the video
capture = cv2.VideoCapture('video.mp4')
counter_line_pos = 550
offset = 6          # allowable error between pixels
vehicle_counter = 0
min_width_rect = 80  # the min width of the rectangle drawn on vehicles
min_height_rect = 80  # the min height of the rectangle drawn on vehicles

# Initializing Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

detected_vehicle = []


def get_center_of_rect(x, y, w, h):
    return (x + int(w/2)), (y + int(h/2))


while True:
    ok, frame = capture.read()

    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 5)
    subtracted_frame = algo.apply(blur)
    dilate_frame = cv2.dilate(subtracted_frame, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_data = cv2.morphologyEx(dilate_frame, cv2.MORPH_CLOSE, kernel)
    dilate_data = cv2.morphologyEx(dilate_data, cv2.MORPH_CLOSE, kernel)

    contourShape, h = cv2.findContours(dilate_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame, (25, counter_line_pos), (1200, counter_line_pos), (0, 0, 255), 3)

    for (idx, contour) in enumerate(contourShape):
        (x, y, w, h) = cv2.boundingRect(contour)
        valid_contour = (w > min_width_rect) and (h > min_height_rect)

        if not valid_contour:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Vehicle: " + str(vehicle_counter), (x, y-20),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)
        center = get_center_of_rect(x, y, w, h)
        detected_vehicle.append(center)
        cv2.circle(frame, center, 4, (0, 0, 255), -1)

        for (cx, cy) in detected_vehicle:
            if (counter_line_pos + offset) > cy > (counter_line_pos - offset):
                vehicle_counter += 1
            cv2.line(frame, (25, counter_line_pos),
                     (1200, counter_line_pos), (0, 127, 255), 3)
            detected_vehicle.remove((cx, cy))
            print("Vehicle Counter : " + str(vehicle_counter))

    cv2.putText(frame, "VEHICLE COUNTER: " + str(vehicle_counter),
                (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    cv2.imshow('Original Input Video', frame)

    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
capture.release()
