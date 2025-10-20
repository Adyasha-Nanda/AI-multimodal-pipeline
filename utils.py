# utils.py
import cv2

def crop_region(frame, box, pad=2):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2]

def draw_detections(frame, detections, ocr_texts):
    for i, (x1, y1, x2, y2, conf, name) in enumerate(detections):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if ocr_texts and i in ocr_texts:
            text_list = ocr_texts[i]
            for j, (text, conf) in enumerate(text_list[:2]):
                cv2.putText(frame, f"{text} ({conf:.2f})",
                            (x1, y2 + 15 + j*15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255,255,0), 1)
    return frame
