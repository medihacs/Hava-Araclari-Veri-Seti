from roboflow import Roboflow
from ultralytics import YOLO

def main():

    rf = Roboflow(api_key="rWQkrN0Q7HK4LS8VAOPo")
    project = rf.workspace("mediha-oyget").project("hava_araclari_tespit-fjxzq")
    version = project.version(1)
    dataset = version.download("yolov8")
    
    print(f"Veri seti şuraya indi: {dataset.location}")

    model = YOLO('yolov8n.pt')  
    
    results = model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=50,     
        imgsz=640,      
        batch=32,       
        name='hava_araclari_modeli', 
        device=0        # GPU:0, CPU=cpu
    )
    
if __name__ == '__main__':
    main()