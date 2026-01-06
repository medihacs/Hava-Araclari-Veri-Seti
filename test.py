from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "yolov8n.pt"



def model_kontrol():
    if not os.path.exists(MODEL_PATH):
        print(f"HATA: '{MODEL_PATH}' bulunamadı!")
        print("Önce modeli eğittiğinden emin ol.")
        return None
    return YOLO(MODEL_PATH)


# ------------------ RESİM TEST ------------------
import os
from ultralytics import YOLO

def image_test():
    model = model_kontrol()
    if model is None:
        return

    # test.py'nin bulunduğu dizin
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # image klasörünün tam yolu
    image_dir = os.path.join(BASE_DIR, "image")

    images = [
        os.path.join(image_dir, "ucak.jpg"),
        os.path.join(image_dir, "iha.jpg"),
        os.path.join(image_dir, "drone.jpg")
    ]

    # Dosya kontrolü (debug)
    for img in images:
        if not os.path.exists(img):
            print(f"❌ Bulunamadı: {img}")
            return

    print("📷 Resim tahmini yapılıyor...")

    model.predict(
        source=images,
        conf=0.40,
        show=False,
        save=True
    )

    print("✅ Tahmin tamamlandı")
    print("📂 Sonuçlar: runs/detect/predict/")




# ------------------ KAMERA / VİDEO TEST ------------------
def camera_test():
    model = model_kontrol()
    if model is None:
        return

    print("🎥 Kamera açılıyor (Çıkmak için 'q')")

    results = model.predict(
        source=0,        # Laptop kamerası
        conf=0.45,
        show=True,
        stream=True
    )

    for r in results:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# ------------------ ANA MENÜ ------------------
def main():
    print("""
==============================
 YOLOv8 Hava Araçları TEST
==============================
1 - Resim Testi
2 - Canlı Kamera Testi
3 - Çıkış
""")

    secim = input("Seçiminizi girin (1/2/3): ")

    if secim == "1":
        image_test()
    elif secim == "2":
        camera_test()
    elif secim == "3":
        print("Çıkılıyor...")
    else:
        print("Geçersiz seçim!")


if __name__ == "__main__":
    main()
