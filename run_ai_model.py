import numpy as np
from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics
import cv2

class YOLO(Model):
    def __init__(self):
        super().__init__(
            model_file="best_imx_model_yolo8v2/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )
        self.labels = np.genfromtxt(
            "best_imx_model/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)

device = AiCamera(image_size=(640, 480), frame_rate=16)
model = YOLO()
device.deploy(model)
annotator = Annotator()

# HSV Thresholds
LOWER_BOUND = np.array([10, 0, 0])
UPPER_BOUND = np.array([45, 255, 255])

with device as stream:
    for frame in stream:
        # --- STEP 1: CAPTURE CLEAN FEED ---
        # We copy the image immediately so it stays "clean" for thresholding
        clean_img = frame.image.copy()

        # --- STEP 2: AI ANNOTATION ---
        # This modifies 'frame.image' directly
        detections = frame.detections[frame.detections.confidence > 0.0]
        labels = [f"{model.labels[class_id]}: {score:0.2f}" for _, score, class_id, _ in detections]
        annotator.annotate_boxes(frame, detections, labels=labels, alpha=0.3, corner_radius=10)
        
        # --- STEP 3: THRESHOLDING ON CLEAN FEED ---
        # Crop the CLEAN image (no boxes or text)
        roi = clean_img[320:480, 0:640] 
        hsv_img = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        binary_view = cv2.inRange(hsv_img, LOWER_BOUND, UPPER_BOUND)

        # --- STEP 4: DISPLAY ---
        # Main window shows AI detections
        cv2.imshow("AI Detections (Annotated)", frame.image)
        # Second window shows the binary result derived from the clean feed
        cv2.imshow("Lane Threshold (Clean Feed)", binary_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
