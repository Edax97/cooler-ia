import os
import time

import picamera2
import numpy as np
import cv2 as cv

from classification.svm_classifier import predict_classes, load_svm, new_cnn_extractor
from segmentation.segmentation import new_model, get_bottles

def alert(_image: np.ndarray, _bottles_detected: list[np.ndarray], _classes_detected: list[str]):
    # save date.alert with <class_detected>, as well as date_image.jpg and date_cropped.jpg
    # dir date/
    date = time.strftime("%Y%m%d-%H%M%S")
    dir_name =f"alerts/{date}"
    os.mkdir(dir_name)
    cv.imwrite(f"{dir_name}/captura.jpg", _image)
    for i, class_str in enumerate(_classes_detected):
        cv.imwrite(f"{dir_name}/item_{i}_{class_str}.jpg", _bottles_detected[i])
        print(f"Detectada clase no permitida: {class_str}")

def get_allowed(filename: str):
    allowed = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        allowed = [l.strip("\r\n") for l in lines]
    return allowed

model_segmentation = new_model()
cnn_extractor = new_cnn_extractor()
svm = load_svm()

def detect_forbidden(cam: str | int=0, is_raspberry=True):
    if is_raspberry:
        picam = picamera2.Picamera2()
        config = picam.create_video_configuration(
            main={"size": (640, 480), "format": "BGR888"}
        )
        picam.configure(config)
        picam.start()
        image = picam.capture_array()
        image = cv.cvtColor(image, cv.COLOR_BGRA2BGR)
    else:
        cap = cv.VideoCapture(cam)
        if not cap.isOpened():
            print("can't open camera")
            exit(1)
        ok, image = cap.read()
        if not ok:
            print("can't read camera")
            exit(1)

    bottles = get_bottles(model_segmentation, image, save=True)
    if len(bottles) == 0:
        print("no bottles detected")
        return

    _allowed = get_allowed("permitidas.txt")

    bottles_classified = predict_classes(svm, cnn_extractor, bottles)

    classes_wrong = []
    bottles_wrong = []
    for j, c in enumerate(bottles_classified):
        if c in _allowed:
            continue
        classes_wrong.append(c)
        b = bottles[j]
        bottles_wrong.append(b.copy())

    if len(classes_wrong) > 0:
        alert(image.copy(),
              _bottles_detected=bottles_wrong,
              _classes_detected=classes_wrong
        )

if __name__ == "__main__":
    detect_forbidden("videos/botellas_1.mp4")