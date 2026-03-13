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


# --- 2. DEFINE THE EMERGENCY STOP FUNCTION ---
def stop_motors():
    print("\n[SHUTDOWN] Stopping motors...")
    if arduino and arduino.is_open:
        # Send zero speed to both motors before closing
        arduino.write(b"0,0\n")
        arduino.close()


# Register the function to run automatically when the script exits (e.g., pressing Ctrl+C)
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

    LOWER_BOUND = np.array(
        [20, 100, 100]
    )  # H: Yellow, S: At least moderately colored, V: At least moderately bright
    UPPER_BOUND = np.array([35, 255, 255])

    # --- PID SETUP ---
    # Kp: Proportional gain (Steering aggressiveness)
    # Ki: Integral gain (Corrects long-term drift)
    # Kd: Derivative gain (Dampens oscillations/wobble)
    Kp = 0.06  # Drastically lowered to stop the violent overcorrecting
    Ki = 0.0
    Kd = 0.0  # Increased to add more "dampening" to the steering

    base_speed = 60  # Slightly lower base speed to handle corners better
    prev_error = 0
    integral = 0

    TARGET_X = 320  # Keep it tracking the exact center

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

            # Thresholding
            roi = clean_img[340:460, 0:640]

            blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)
            hsv_img = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2HSV)
            binary_view = cv2.inRange(hsv_img, LOWER_BOUND, UPPER_BOUND)
            kernel = np.ones((5, 5), np.uint8)
            binary_view = cv2.morphologyEx(binary_view, cv2.MORPH_OPEN, kernel)
            binary_view = cv2.morphologyEx(binary_view, cv2.MORPH_CLOSE, kernel)

            M = cv2.moments(binary_view)
            # print(f"Moments: {M}")

            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - TARGET_X

                integral += error
                derivative = error - prev_error
                prev_error = error

                turn = (Kp * error) + (Ki * integral) + (Kd * derivative)

                # --- 2. DYNAMIC CORNERING SPEED ---
                # Slow down the forward speed proportionally to how sharp the turn is.
                # E.g., if error is 150 (sharp turn), base speed drops by 22.5
                current_speed = base_speed - (abs(error) * 0.05)

                # Make sure the car doesn't completely stall out (keep minimum speed at 20)
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

            # Convert binary image (1 channel) to BGR (3 channels) so we can stack it
            binary_3ch = cv2.cvtColor(binary_view, cv2.COLOR_GRAY2BGR)

            # Stack images vertically (Main Feed on top, Threshold on bottom)
            combined_view = cv2.vconcat([frame.image, binary_3ch])

            # Compress to JPEG
            ret, buffer = cv2.imencode(".jpg", combined_view)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Yield the frame to the web server
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )


# --- 5. THE WEB ROUTE ---
@app.route("/")
def video_feed():
    # This route tells the browser to expect a continuous stream of JPEGs
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# --- 6. GET IP AND RUN ---
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

    # Run the server on port 5000
    app.run(host="0.0.0.0", port=5000, threaded=True)
