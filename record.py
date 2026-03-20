import numpy as np
import cv2
import serial
import atexit
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model

# --- 1. SET UP SERIAL & SAFETY SHUTDOWN ---
arduino = None
try:
    arduino = serial.Serial("/dev/ttyACM0", 9600, timeout=0.1)
except Exception as e:
    print(f"Serial connection failed: {e}")


def stop_motors():
    print("\n[SHUTDOWN] Stopping motors...")
    if arduino and arduino.is_open:
        arduino.write(b"0,0\n")
        arduino.close()


atexit.register(stop_motors)


# --- 2. LOCAL DUMMY MODEL SETUP ---
# We load your existing local model just to satisfy the camera's requirement,
# completely bypassing the need for an internet connection.
class LocalDummyModel(Model):
    def __init__(self):
        super().__init__(
            model_file="best_imx_model_yolo8v2/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

    def post_process(self, output_tensors):
        return []


def main():
    # --- 3. SET UP VIDEO RECORDER ---
    frame_width, frame_height = 640, 480
    fps = 16.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("robot_run.mp4", fourcc, fps, (frame_width, frame_height))

    # --- 4. VISION & PID SETTINGS ---
    LOWER_YELLOW = np.array([20, 100, 100])
    UPPER_YELLOW = np.array([35, 255, 255])

    LOWER_WHITE = np.array([0, 0, 180])
    UPPER_WHITE = np.array([180, 50, 255])

    # Settings synced with run_robot.py
    Kp = 0.2
    Ki = 0.0
    Kd = 0.1

    base_speed = 80
    TARGET_X = 320

    prev_error = 0
    integral = 0

    # --- MEMORY VARIABLE ---
    yellow_is_right = True
    LANE_WIDTH_ESTIMATE = 450

    print("\n" + "=" * 40)
    print("🚗 OFFLINE AI RECORDING LINE FOLLOWER IS RUNNING!")
    print("🎥 Recording to 'robot_run.mp4'...")
    print("🛑 Press Ctrl+C in the terminal to stop.")
    print("=" * 40 + "\n")

    # --- 5. INITIALIZE AI CAMERA & DEPLOY LOCAL MODEL ---
    device = AiCamera(image_size=(frame_width, frame_height), frame_rate=int(fps))
    model = LocalDummyModel()
    device.deploy(model)  # Prevents the online download

    try:
        with device as stream:
            for frame in stream:
                clean_img = frame.image.copy()

                # Write the raw, clean frame to the video file
                out.write(clean_img)

                # --- 6. IMAGE PROCESSING ---
                roi = clean_img[300:460, 0:640]
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                hsv_img = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

                mask_yellow = cv2.inRange(hsv_img, LOWER_YELLOW, UPPER_YELLOW)
                mask_white = cv2.inRange(hsv_img, LOWER_WHITE, UPPER_WHITE)

                kernel = np.ones((5, 5), np.uint8)
                mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
                mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

                # --- 7. INDEPENDENT COLOR TRACKING ---
                M_y = cv2.moments(mask_yellow)
                M_w = cv2.moments(mask_white)

                cx_y = int(M_y["m10"] / M_y["m00"]) if M_y["m00"] > 0 else None
                cx_w = int(M_w["m10"] / M_w["m00"]) if M_w["m00"] > 0 else None

                # --- 8. SMART LANE ESTIMATION ---
                if cx_y is not None and cx_w is not None:
                    # We can see both lines! Calculate perfect center.
                    lane_center = (cx_y + cx_w) // 2

                    # UPDATE MEMORY: Which side is the yellow line on right now?
                    yellow_is_right = cx_y > cx_w

                elif cx_y is not None:
                    # We can only see Yellow (White disappeared on a turn)
                    if yellow_is_right:
                        lane_center = cx_y - (LANE_WIDTH_ESTIMATE // 2)
                    else:
                        lane_center = cx_y + (LANE_WIDTH_ESTIMATE // 2)

                elif cx_w is not None:
                    # We can only see White (Yellow disappeared on a turn)
                    if yellow_is_right:  # If yellow is right, white must be left
                        lane_center = cx_w + (LANE_WIDTH_ESTIMATE // 2)
                    else:
                        lane_center = cx_w - (LANE_WIDTH_ESTIMATE // 2)
                else:
                    lane_center = TARGET_X

                # --- 9. CALCULATE ERROR & PID ---
                error = lane_center - TARGET_X

                integral += error
                derivative = error - prev_error
                prev_error = error

                turn = (Kp * error) + (Ki * integral) + (Kd * derivative)

                # --- 10. CORNER BRAKING ---
                current_speed = max(40, base_speed)

                left_speed = int(current_speed + turn)
                right_speed = int(current_speed - turn)

                left_speed = max(0, min(100, left_speed))
                right_speed = max(0, min(100, right_speed))

                if cx_y is not None or cx_w is not None:
                    command = f"{left_speed},{right_speed}\n".encode()
                else:
                    command = b"0,0\n"
                    integral = 0

                if arduino:
                    arduino.write(command)

    except KeyboardInterrupt:
        print("\n[INFO] Manual interrupt received.")
    finally:
        out.release()
        print("[INFO] Video saved successfully as 'robot_run.mp4'.")


if __name__ == "__main__":
    main()
