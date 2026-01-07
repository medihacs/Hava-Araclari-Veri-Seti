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
        os.path.join(image_dir, "iha2.jpg")
    ]

    # Dosya kontrolü (debug)
    for img in images:
        if not os.path.exists(img):
            print(f"Bulunamadı: {img}")
            return

    print("📷 Resim tahmini yapılıyor...")

    model.predict(
        source=images,
        conf=0.40,
        show=False,
        save=True
    )

    print("Tahmin tamamlandı")
    print("Sonuçlar: runs/detect/predict/")




# ------------------  VİDEO TEST ------------------
def video_test():

    model = model_kontrol()
    if model is None:
        return

    video_paths = ["video/drone.mp4" ,
                  "video/ucak.mp4"
      ]  # video klasöründeki dosyalar

    # 2. Dosyaların Varlığını TEK TEK Kontrol Et
    gecerli_videolar = [] # Sadece gerçekten var olanları buraya ekleyeceğiz
    
    for video in video_paths:
        if os.path.exists(video):
            gecerli_videolar.append(video)
        else:
            print(f"⚠️ UYARI: Video bulunamadı ve atlanacak: {video}")

    # Eğer hiç geçerli video yoksa işlemi durdur
    if not gecerli_videolar:
        print("Hiçbir video dosyası bulunamadı!")
        return

    for video_dosyasi in gecerli_videolar:
        print(f"İşleniyor: {video_dosyasi}")
        
        try:
            model.predict(
                source=video_dosyasi, # Buraya LİSTE değil, TEK dosya veriyoruz
                conf=0.45,
                show=False,
                save=True
            )
        except Exception as e:
            print(f"⚠️ Hata oluştu ({video_dosyasi}): {e}")

    print("Video tahmini tamamlandı")
    print("Sonuçlar: runs/detect/predict/")

# ------------------ KAMERA  TEST ------------------
def camera_test():
    model = model_kontrol()
    if model is None:
        return

    print("Kamera açılıyor (Çıkmak için 'q')")

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
3 - Video Testi      
4 - Çıkış
""")

    secim = input("Seçiminizi girin (1/2/3): ")

    if secim == "1":
        image_test()
    elif secim == "2":
        camera_test()
    elif secim == "3":
        video_test()
    elif secim == "4":
        print("Çıkılıyor...")
    else:
        print("Geçersiz seçim!")


if __name__ == "__main__":
    main()

