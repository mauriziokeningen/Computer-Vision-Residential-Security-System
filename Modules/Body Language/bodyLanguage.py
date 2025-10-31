import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import time

result_list=[]

#callback function to process results
def res_callback(result, output, timestamp_ms):
    result_list.append(result)

#specify config
options = vision.PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path='Modules/Body Language/model/pose_landmarker_lite.task'),
    running_mode = vision.RunningMode.LIVE_STREAM,
    result_callback = res_callback)
landmarker = vision.PoseLandmarker.create_from_options(options)

#read input video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret == False:
        break

    h,w,_ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame_rgb)

    #get results
    landmarker.detect_async(frame_rgb, time.time_ns() // 1_000_000)

    if result_list:
        for lm in result_list[0].pose_landmarks:
            for each_lm in lm:
                if each_lm.visibility >0.9:
                    x_each_lm = int(each_lm.x*w)
                    y_each_lm = int(each_lm.y * h)
                    cv2.circle(frame, (x_each_lm, y_each_lm), 3, (0,255,255), -1)
        result_list.clear()
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF ==27:
        break

cap.release()
cv2.destroyAllWindows()

