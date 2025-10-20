# inference.py
import cv2
import psutil
import time
from detector import Detector
from ocr_reader import OCRReader
from utils import crop_region, draw_detections

def main(source=0):
    cap = cv2.VideoCapture(source)
    detector = Detector()
    ocr = OCRReader()

    frame_count = 0
    start_time = time.time()
    cpu_usage = []
    mem_usage = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.predict(frame)
        ocr_texts = {}

        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf, name = det
            crop = crop_region(frame, (x1, y1, x2, y2))
            texts = ocr.read_text(crop)
            ocr_texts[i] = texts

        output_frame = draw_detections(frame, detections, ocr_texts)
        cv2.imshow("Multimodal Pipeline", output_frame)

        frame_count += 1
        cpu_usage.append(psutil.cpu_percent())
        mem_usage.append(psutil.virtual_memory().percent)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0
    print(f"✅ Average FPS: {fps:.2f}")
    print(f"📊 Avg CPU usage: {sum(cpu_usage)/len(cpu_usage):.1f}%")
    print(f"💾 Avg RAM usage: {sum(mem_usage)/len(mem_usage):.1f}%")

if __name__ == "__main__":
    main(0)  # 0 = webcam, or replace with 'path/to/image.jpg'
