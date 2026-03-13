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
        # The library requires this function to exist.
        # We don't care about detections, so we just return an empty list.
        return []


def main():
    # --- 3. SET UP VIDEO RECORDER ---
    frame_width, frame_height = 640, 480
    fps = 16.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("robot_run.mp4", fourcc, fps, (frame_width, frame_height))

    # --- 4. VISION & PID SETTINGS ---
    LOWER_BOUND = np.array([20, 100, 100])
    UPPER_BOUND = np.array([35, 255, 255])

    Kp = 0.06
    Ki = 0.0
    Kd = 0.0
    base_speed = 60
    TARGET_X = 320

    prev_error = 0
    integral = 0

    print("\n" + "=" * 40)
    print("🚗 OFFLINE AI CAMERA LINE FOLLOWER IS RUNNING!")
    print("🎥 Recording to 'robot_run.mp4'...")
    print("🛑 Press Ctrl+C in the terminal to stop.")
    print("=" * 40 + "\n")

    # --- 5. INITIALIZE AI CAMERA & DEPLOY LOCAL MODEL ---
    device = AiCamera(image_size=(frame_width, frame_height), frame_rate=int(fps))
    model = LocalDummyModel()
    device.deploy(model)  # This prevents the online download

    try:
        with device as stream:
            for frame in stream:
                clean_img = frame.image.copy()

                # Write the clean frame to the video file
                out.write(clean_img)

                # --- 6. IMAGE PROCESSING ---
                roi = clean_img[340:460, 0:640]
                blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
                hsv_img = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
                binary_view = cv2.inRange(hsv_img, LOWER_BOUND, UPPER_BOUND)

                kernel = np.ones((5, 5), np.uint8)
                binary_view = cv2.morphologyEx(binary_view, cv2.MORPH_OPEN, kernel)
                binary_view = cv2.morphologyEx(binary_view, cv2.MORPH_CLOSE, kernel)

                # --- 7. CALCULATE MOMENTS & PID ---
                M = cv2.moments(binary_view)

                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    error = cx - TARGET_X

                    integral += error
                    derivative = error - prev_error
                    prev_error = error

                    turn = (Kp * error) + (Ki * integral) + (Kd * derivative)

                    current_speed = base_speed - (abs(error) * 0.05)
                    current_speed = max(60, current_speed)

                    left_speed = int(current_speed + turn)
                    right_speed = int(current_speed - turn)

                    left_speed = max(0, min(100, left_speed))
                    right_speed = max(0, min(100, right_speed))

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
