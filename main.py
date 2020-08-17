import cv2
import dlib
import numpy as np
import pafy

concentration = 100
con_list = []
face_frequency = []
eye_center_frequency = []
eye_right_frequency = []
eye_left_frequency = []

time_count = 0
time_flag = 0
blue = 0
green = 128
red = 0
url = "https://www.youtube.com/watch?v=PPLop4L2eGk"
youtube_video = pafy.new(url)
best = youtube_video.getbest(preftype='mp4')
cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(best.url)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


font = cv2.FONT_HERSHEY_PLAIN


def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)],
                               np.int32)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y: max_y, min_x: max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 2
    else:
        gaze_ratio = left_side_white / right_side_white
    return gaze_ratio


while True:
    time_count += 1
    _, frame = cap.read()
    _, frame2 = cap2.read()  # youtube
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if time_flag == 0:
        face_frequency.append(len(faces))
        time_flag = 1
        if face_frequency[-1] == 0:
            concentration -= 0.25
        con_list.append(concentration)

    # if time_flag and 0.9 < abs(second_time - first_time) < 1.1:
    if time_flag and time_count == 21:
        face_frequency.append(len(faces))
        time_count = 0

        if face_frequency[-1] == 0:
            concentration -= 1
            if concentration <= 0:
                concentration = 0
        else:
            concentration += 0.25
            if concentration >= 100:
                concentration = 100
        con_list.append(concentration)

    if 50 <= concentration <= 100:
        red = (100 - concentration) * 5
        green = 128 + (100 - concentration)
    else:
        red = 250
        green = concentration * 3

    frame = cv2.line(frame, (600, 600), (600, 4 * int(100 - concentration)), (blue, green, red), 20)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Gaze detection
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

        # print(gaze_ratio)
        if gaze_ratio <= 1.0:
            cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
        elif 1.0 < gaze_ratio < 2.0:
            cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)

    cv2.imshow("Frame", frame)
    cv2.imshow('youtube', frame2)

    if cv2.waitKey(1) == 27:
        break

worst_cnt = 0
worst_max = 0
worst_time_first = 0
worst_time_second = 0
first_flag = 1
worst_time_first_final = 0

# calculate worst concentration period
for i in range(len(face_frequency) - 1):
    if face_frequency[i] == 0 and face_frequency[i + 1] == 0:
        if first_flag:
            worst_time_first = i
            first_flag = 0
        worst_cnt += 1
    else:
        if worst_max < worst_cnt:
            worst_max = worst_cnt
            worst_time_first_final = worst_time_first
            worst_time_second = i
        first_flag = 1
        worst_cnt = 0

print("worst_time_first: ", worst_time_first_final)
print("worst_time_second: ", worst_time_second)
print(face_frequency)
# print(len(face_frequency))
print(con_list)
# print(len(con_list))
cap.release()
cap2.release()
cv2.destroyAllWindows()

with open('list_file.txt', 'w') as f:
    for item in con_list:
        f.write("%s\n" % item)

