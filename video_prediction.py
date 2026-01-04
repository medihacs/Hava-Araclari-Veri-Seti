from ultralytics import YOLO
import cv2

def canli_test():
    # 1. Modeli Yükle
    model_path = 'runs/weights/best.pt'
    model = YOLO(model_path)

    # 2. Kamerayı Aç (0 genellikle laptop kamerasidir)
    results = model.predict(
        source=0, 
        show=True,    # Canlı pencere aç
        conf=0.45,    # Güven eşiği
        stream=True   # Sürekli akış (video/kamera için şart)
    )
    
    # Not: Çıkmak için açılan penceredeyken 'q' tuşuna basabilirsin.
    for r in results:
        pass # Stream modu için döngü gereklidir

if __name__ == '__main__':
    canli_test()

