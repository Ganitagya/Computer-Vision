# Importing the libraries
import cv2
import time


def get_center_of_rect(x, y, w, h):
    return (x + int(w / 2)), (y + int(h / 2))


def main():
    # Loading the cascade
    cars_cascade = cv2.CascadeClassifier('../Haarcascade/haarcascade_cars.xml')
    # capture = cv2.VideoCapture('dataset/video.mp4')

    capture = cv2.VideoCapture('SheikhZayed.mp4')
    counter_line_pos = 550
    offset = 6  # allowable error between pixels
    vehicle_counter = 0
    min_width_rect = 80  # the min width of the rectangle drawn on vehicles
    min_height_rect = 80  # the min height of the rectangle drawn on vehicles

    detected_vehicle = []

    while True:
        time.sleep(0.5)
        ok, frame = capture.read()
        cv2.line(frame, (25, counter_line_pos), (1200, counter_line_pos), (0, 0, 255), 3)

        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, 1.16, 1)

        for (x, y, w, h) in cars:
            center = get_center_of_rect(x, y, w, h)
            detected_vehicle.append(center)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

        # Split the image into its color channels
        b, g, r = cv2.split(frame)

        # Threshold the red channel to create a binary image: as all our rectangles are red
        _, binary = cv2.threshold(r, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours
        for contour in contours:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # If the contour has 4 corners, it is likely a rectangle
            # If not then ignore and move on
            if not len(approx) == 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            valid_contour = (w > min_width_rect) and (h > min_height_rect)

            if not valid_contour:
                continue

            for (cx, cy) in detected_vehicle:
                if (counter_line_pos + offset) > cy > (counter_line_pos - offset):
                    vehicle_counter += 1
                    print("Vehicle Counter : " + str(vehicle_counter))
                cv2.line(frame, (25, counter_line_pos),
                         (1200, counter_line_pos), (0, 127, 255), 3)
                detected_vehicle.remove((cx, cy))

        cv2.putText(frame, "VEHICLE COUNTER: " + str(vehicle_counter),
                    (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 127), 3)

        cv2.imshow('Original Input Video', frame)

        if cv2.waitKey(1) == 13:
            break

    cv2.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()
