# ocr_reader.py
import easyocr
import cv2

class OCRReader:
    def __init__(self, langs=['en']):
        print("🔹 Initializing EasyOCR...")
        self.reader = easyocr.Reader(langs, gpu=False)

    def read_text(self, image_crop):
        rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb)
        texts = []
        for (bbox, text, conf) in results:
            texts.append((text, conf))
        return texts
