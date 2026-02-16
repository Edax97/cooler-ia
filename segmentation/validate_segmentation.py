from ultralytics import YOLO

CONF=0.2
IOU=0.60
CLASSES = [39,41,76]
def validate_fridges(model_, image_dir_: str) -> int:
    results = model_.predict(image_dir_, conf=CONF, iou=IOU, classes=CLASSES, save=True)
    segs_total = 0
    for j, r in enumerate(results):
        if r.masks:
            segs_total += len(r.masks)
    return segs_total

if __name__ == "__main__":
    models = ["yolo26m-seg.pt", "yolo26n-seg.pt"]
    image_dir = "../data/situaciones"
    for m in models:
        model = YOLO(m)
        detected = validate_fridges(model, image_dir)
        print(f"Model {m}: {detected}")
