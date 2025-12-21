# Hava Araçları Veri Seti
Bu veri seti 10.825 adet etiketli uçak, drone ve İHA görseli içermektedir.

## Veriyi İndirmek İçin Python Kodu:
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="rWQkrN0Q7HK4LS8VAOPo")
project = rf.workspace("mediha-oyget").project("hava_araclari_tespit-fjxzq")
version = project.version(1)
dataset = version.download("yolov8")
                
