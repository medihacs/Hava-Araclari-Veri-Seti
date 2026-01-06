from ultralytics import YOLO
import cv2
import os

model_path = 'runs/weights/best.pt'



def model_kontrol():
    if not os.path.exists(model_path):
        print(f"HATA: '{model_path}' bulunamadı!")
        print("Önce modeli eğittiğinden emin ol.")
        return None
    return YOLO(model_path)


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
        os.path.join(image_dir, "drone.jpg"),
        os.path.join(image_dir, "drone2.jpg"),
        os.path.join(image_dir, "ucak2.jpg"),
        os.path.join(image_dir, "iha2.jpg"),
        os.path.join(image_dir, "kus.jpg")

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




# ------------------  VİDEO TEST ------------------
def video_test():
    print("🎥 Video tahmini yapılıyor...")

    model = model_kontrol()
    if model is None:
        return

    video_path = "video/drone.mp4"   # video klasöründeki dosya

    if not os.path.exists(video_path):
        print(f"❌ Video bulunamadı: {video_path}")
        return

    model.predict(
        source=video_path,
        conf=0.45,
        show=False,   # ❌ pencere açma (hata almamak için)
        save=True     # ✅ sonucu kaydet
    )

    print("✅ Video tahmini tamamlandı")
    print("📂 Sonuçlar: runs/detect/predict/")

# ------------------ KAMERA  TEST ------------------
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
2 - Video Testi
3 - Canlı Kamera Testi      
4 - Çıkış
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
