import numpy as np
import cv2
import socket
import serial
import atexit
from flask import Flask, Response
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics

# --- 1. SET UP THE WEB SERVER ---
app = Flask(__name__)

# --- GLOBAL SERIAL PORT FOR SAFETY SHUTDOWN ---
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


# --- 3. YOUR AI MODEL ---
class YOLO(Model):
    def __init__(self):
        super().__init__(
            model_file="best_imx_model_yolo8v2/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(
            "best_imx_model_yolo8v2/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)


# --- 4. THE VIDEO STREAM GENERATOR ---
def generate_frames():
    device = AiCamera(image_size=(640, 480), frame_rate=16)
    model = YOLO()
    device.deploy(model)
    annotator = Annotator()

    # --- VISION SETTINGS ---
    LOWER_YELLOW = np.array([20, 100, 100])
    UPPER_YELLOW = np.array([35, 255, 255])

    LOWER_WHITE = np.array([0, 0, 180])
    UPPER_WHITE = np.array([180, 50, 255])

    # --- PID SETUP (Tuned for sharper turns!) ---
    Kp = 0.1  # Increased from 0.06 so it steers much harder into corners
    Ki = 0.0
    Kd = 0.05  # Added a little Kd to stop it from wobbling as it exits a turn

    base_speed = 100
    prev_error = 0
    integral = 0

    TARGET_X = 320

    # --- MEMORY VARIABLE ---
    # The robot will learn this automatically as it drives
    yellow_is_right = True
    LANE_WIDTH_ESTIMATE = 450

    with device as stream:
        for frame in stream:
            clean_img = frame.image.copy()

            # AI Annotation
            detections = frame.detections[frame.detections.confidence > 0.0]
            labels = [
                f"{model.labels[class_id]}: {score:0.2f}"
                for _, score, class_id, _ in detections
            ]
            annotator.annotate_boxes(
                frame, detections, labels=labels, alpha=0.3, corner_radius=10
            )

            # --- IMAGE PROCESSING (Now looking further ahead!) ---
            # Raised ROI from 340 to 300 so it can see corners coming sooner
            roi = clean_img[300:460, 0:640]
            blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv_img = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)

            mask_yellow = cv2.inRange(hsv_img, LOWER_YELLOW, UPPER_YELLOW)
            mask_white = cv2.inRange(hsv_img, LOWER_WHITE, UPPER_WHITE)

            kernel = np.ones((5, 5), np.uint8)
            mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
            mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

            # --- INDEPENDENT COLOR TRACKING ---
            M_y = cv2.moments(mask_yellow)
            M_w = cv2.moments(mask_white)

            cx_y = int(M_y["m10"] / M_y["m00"]) if M_y["m00"] > 0 else None
            cx_w = int(M_w["m10"] / M_w["m00"]) if M_w["m00"] > 0 else None

            # --- SMART LANE ESTIMATION ---
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

            # --- CALCULATE ERROR & PID ---
            error = lane_center - TARGET_X

            integral += error
            derivative = error - prev_error
            prev_error = error

            turn = (Kp * error) + (Ki * integral) + (Kd * derivative)

            # --- CORNER BRAKING ---
            # Increased penalty from 0.05 to 0.15 so it brakes harder for sharp turns
            current_speed = base_speed - (abs(error) * 0.15)
            current_speed = max(
                40, current_speed
            )  # Allowed it to slow down to 40 instead of 60

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

            # Web Feed Visualization
            binary_view = cv2.bitwise_or(mask_yellow, mask_white)
            binary_3ch = cv2.cvtColor(binary_view, cv2.COLOR_GRAY2BGR)
            combined_view = cv2.vconcat([frame.image, binary_3ch])

            ret, buffer = cv2.imencode(".jpg", combined_view)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


# --- 5. THE WEB ROUTE ---
@app.route("/")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"
    finally:
        s.close()
    return IP


if __name__ == "__main__":
    pi_ip = get_ip()
    print("\n" + "=" * 50)
    print("🚀 AI STREAM IS LIVE!")
    print(f"👉 Click here or paste this into your browser: http://{pi_ip}:5000")
    print("=" * 50 + "\n")

    app.run(host="0.0.0.0", port=5000, threaded=True)
