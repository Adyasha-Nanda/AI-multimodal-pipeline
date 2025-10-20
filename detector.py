# detector.py
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path='yolov8n.pt', conf=0.25, device='cpu'):
        print("🔹 Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def predict(self, frame):
        results = self.model.predict(
            source=frame, conf=self.conf, device=self.device, verbose=False
        )
        detections = []
        if len(results) == 0:
            return detections
        r = results[0]
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = self.model.names[cls]
            detections.append((x1, y1, x2, y2, conf, name))
        return detections
