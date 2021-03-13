from yolov4.tf import YOLOv4
from collections import Counter
from PIL import Image
import numpy as np
import psutil
import sys

yolo = YOLOv4()
yolo = YOLOv4(tiny=True)

yolo.classes = "/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Demo_Python/cocodata/custom.names"
# yolo.classes = str(sys.argv[1])

# print("Label values", yolo.classes)
print("Label values", yolo.classes[0])
# print("Label values", yolo.classes[1])

yolo.input_size = (640, 480)
print("Toi day roi ne")

yolo.make_model()
yolo.load_weights("/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/Demo_Python/weight/yolov4-tiny-final.weights", weights_type="yolo")
# yolo.load_weights("yolov4-tiny.weights", weights_type="yolo")

im = Image.open("/Users/truongtn/Desktop/homies1.jpg")
im = np.array(im)

result = yolo.predict(im)
yolo.inference(media_path="/Users/truongtn/Desktop/homies1.jpg")

# yolo.save_as_tflite("/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/DemoYolov4_v1/yolov4.tflite")

rows = len(result)
columns = len(result[0])

print("Row size: ", rows)
print("Column size: ", columns)
print("Result: ", result.tolist())
# print("Result: ", yolo.classes[result[0][4]])

# yolo.inference(media_path="/Users/truongtn/Desktop/homies1.jpg")


# yolo.inference(media_path="/Users/truongtn/Desktop/Desktop/HocTap/Semester8/Python/tensorflow-yolov4-master/test/road.mp4", is_image=False)
