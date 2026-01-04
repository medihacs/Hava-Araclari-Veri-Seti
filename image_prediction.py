from ultralytics import YOLO
import cv2

def tahmin_et():
    model_path = 'runs/weights/best.pt'
    
    try:
        model = YOLO(model_path)
    except:
        print(f"HATA: '{model_path}' bulunamadı. Eğitimin bittiğinden emin misin?")
        return

    source_path = ["ucak.jpg","iha.jpg","drone.jpg"] 

    # 3. Tahmin Yap
    print("Tahmin yapılıyor...")
    results = model.predict(
        source=source_path, 
        conf=0.40,     
        save=True,      
        show=True       
    )

if __name__ == '__main__':
    tahmin_et()