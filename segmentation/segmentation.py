import os

import numpy as np
from ultralytics import YOLO
import cv2 as cv


CONF=0.25
IOU=0.30
CLASSES = [39]
def save_imgs(image_list: list[np.ndarray], save_dir: str):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for j, img in enumerate(image_list):
        cv.imwrite(os.path.join(save_dir, f"bottle_{j}.jpg"), img)

def new_model(_model_path="models/yolo26l-seg.pt"):
    return YOLO(_model_path)

def get_bottles(_model: YOLO, _image: np.ndarray, save=False):
    print("get bottles")

    img_h, img_w = _image.shape[:2]

    results = _model.predict(_image, conf=CONF, iou=IOU, classes=CLASSES)

    if len(results) == 0:
        return []

    masks = results[0].masks.xy
    boxes = results[0].boxes.xyxy
    bottles = []
    for j, mask in enumerate(masks):
        x, y, x1, y1 = boxes[j]
        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        masked_bg = np.zeros((img_h, img_w), dtype=np.uint8)
        polygon = np.array(mask, dtype=np.int32)
        cv.fillPoly(masked_bg, [polygon], 255)
        masked_img = cv.bitwise_and(_image, _image, mask=masked_bg)
        bottle_img = masked_img[y:y1, x:x1]
        bottles.append(bottle_img)
    if save:
        save_imgs(bottles, "bottles_masked")
    return bottles

if __name__ == '__main__':
    model_path = "yolo26l-seg.pt"
    image = "../data/situaciones/4-botellas.jpeg"
    model = YOLO(model_path)
    image_mat = cv.imread(image)
    get_bottles(model, image_mat, save=True)
