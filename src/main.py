import threading
import time
import numpy as np
import cv2
import imutils
from adafruit_servokit import ServoKit
import board
import busio
from lane_detection import main as lane_detect
from traffic_sign import main as sign_detect

#--------------------------------Model-----------------------------#

#--------------------------------Model-endl------------------------#

#--------------------------------Biến môi trường-----------------------------#
control_signal = 1
steering_angle = None
road_sign = None

delta = 15  
normal_speed = 0.16
bend_speed = 0.17
error = 0.8
time_sleep = 1
min_bend = 30

cur_frame = None
sign_frame = None
angle = 120
speed = normal_speed

count = 0
time_for_signal = 0.7

is_running = True

#---------------------------------Kết thúc biến môi trường----------------------------#

# Hàm thực hiện phát hiện lane
def detect_lane():
    global cur_frame, speed, angle, control_signal, is_running, delta
    try:
        while is_running:
            if control_signal != 1:
                continue
            if cur_frame is None: 
                continue
            start = time.time()
            angle = lane_detect.steering(cur_frame)
            if (angle >= 120 - delta and angle <= 120 + delta):
                speed = normal_speed
            #else:
             #   esc.throttle = bend_speed
            elif (angle > 60 and angle < (120 - delta)):
                speed =  bend_speed - ((120-angle)/60)*0.03
            elif (angle > (120 + delta) and angle < 180):
                speed = bend_speed
            # print("Lane: ", str(round(time.time() - start, 4)))
    except Exception as e:
        print(f"Lane detection error: {e}")

# Hàm thực hiện phát hiện biển báo
def detect_traffic_sign():
    global control_signal, road_sign, sign_frame, angle, speed, is_running, count, time_for_signal
    prev_sign = None
    cur_sign = None
    Known_distance = 215  # cm
    Known_width = 6  # cm

    distance_road_sign = 0
    ref_image = cv2.imread("frame_1.png")
    ref_image_obj_width = sign_detect.obj_data(ref_image)
    Focal_length_found = sign_detect.Focal_Length_Finder(Known_distance, Known_width, ref_image_obj_width)
    try:
        while is_running:
            # if count % 2 == 0:
            #     continue
            if sign_frame is None: 
                continue
            start = time.time()
            cur_sign = sign_detect.detectStopSign(sign_frame)
            result = sign_detect.findTrafficSign(sign_frame)
            # print("Detect time: ", str(round(time.time() - start, 4)))
            if result is not None:
                detected_sign, sign_position, confidence, distance = result
                if detected_sign is not None:
                    cur_sign = detected_sign
                print("Detected sign:", detected_sign)
            print(f"current sign: {cur_sign}, previous sign: {prev_sign}")
            if cur_sign is None and prev_sign is not None:
                time_for_signal = sign_detect.distance_to_time(distance_road_sign, 50, 0.01, 0.7)
                control_signal = 2
                road_sign = prev_sign
            prev_sign = cur_sign
            sign_frame_resized = imutils.resize(sign_frame, width=1000)
            sign_frame_cropped = sign_frame_resized[90:960, 0:1280]
            obj_width_in_frame = sign_detect.obj_data(sign_frame_cropped)
            if obj_width_in_frame != 0:
                distance_road_sign = sign_detect.Distance_finder(Focal_length_found, Known_width, obj_width_in_frame)
            # print("Road sign: ", str(round(time.time() - start, 4)))
    except Exception as e:
        print(f"Traffic sign detection error: {e}")
# Hàm thực hiện điều khiển
def main():
    print("Initializing Servos")

    global control_signal, is_running, cur_frame, road_sign, sign_frame, count, angle, speed, error, time_sleep
    i2c_bus1 = busio.I2C(board.SCL, board.SDA)
    kit = ServoKit(channels=16, i2c=i2c_bus1, frequency=50)
    esc = kit.continuous_servo[2]
    servo = kit.servo[0]

    lane_thread = threading.Thread(target=detect_lane)
    traffic_sign_thread = threading.Thread(target=detect_traffic_sign)
    lane_thread.start()
    traffic_sign_thread.start()
    try:
        camera = cv2.VideoCapture(0)
        # time.sleep(1.5)
        servo.angle = 120
        esc.throttle = normal_speed
        while camera.isOpened() and is_running:
            count += 1
            ret, frame = camera.read()
            if not ret:
                break
            start = time.time()
            sign_frame = frame.copy()
            cur_frame = frame.copy()
            angle = np.clip(angle, 60, 180)
            print("Control signal: ", control_signal)
            if control_signal == 1:
                pass
                # cv2.putText(frame, str(angle), (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 225, 0), 2)
            elif control_signal == 2:
                print("\n"*10)
                print("Time for steering: ", time_for_signal - error)
                print("\n"*10)
                # 1: left, 2: right, 3: straight, 4: stops
                if road_sign == 4:
                    esc.throttle = 0
                    print("\n"*10)
                    print("stop")
                    print("\n"*10)
                    time.sleep(time_for_signal - 1.5)
                    break
                time.sleep(time_for_signal - error)
                if road_sign == 1:
                    print("Turn left!")
                    speed = bend_speed 
                    angle = 65
                elif road_sign == 2:
                    speed =  bend_speed 
                    angle = 175
                    time.sleep(0.3)
                elif road_sign == 3:
                    control_signal = 1
                    continue
                elif road_sign == 5:
                    esc.throttle = 0
            # out.write(frame)
            esc.throttle = speed
            servo.angle = angle
            if control_signal == 2:
                print("Sleep!")
                time.sleep(time_sleep)
                servo.angle = 120
                control_signal = 1

    except Exception as e:
        print('\n'*10)
        print(f"Control error: {e}")
        esc.throttle = 0
    finally:
        camera.release()
        cv2.destroyAllWindows()
        esc.throttle = 0
main()