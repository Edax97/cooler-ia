import pickle
from pickletools import uint8

import cv2 as cv
import numpy as np
import torch
from PIL import Image
from sklearn.svm import SVC
from torch import nn
from torch.nn import Sequential
from torchvision import models, transforms

index_to_class = {
    0: "7up",
    1: "background",
    2: "cocacola",
    3: "incacola",
    4: "pepsi",
    5: "sprite",
}

SVM_FILE = "../models/svm.pkl"
def save_svm(svm, name=SVM_FILE):
    with open(name, 'wb') as file:
        pickle.dump(svm, file)

def load_svm(name=SVM_FILE):
    file = open(name, 'rb')
    return pickle.load(file)

device = torch.device("cpu")
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# []mat -> []class
def predict_classes(_model_svm: SVC, _cnn_extractor: Sequential, _images: list[np.ndarray]) -> list[str]:
    tensors = []
    for _image in _images:
        _image_rgb = cv.cvtColor(_image, cv.COLOR_BGR2RGB)
        _image_pil = Image.fromarray(_image_rgb)
        _image_tensor = preprocess(_image_pil)
        tensors.append(_image_tensor)

    batch_tensor = torch.stack(tensors)
    embeddings = get_embedding(_cnn_extractor, batch_tensor)

    y = _model_svm.predict(embeddings)
    classes= []
    for i in y.tolist():
        class_str = index_to_class[i]
        classes.append(class_str)
    return classes

def get_embedding(_cnn_extractor: Sequential, batch_input_tensor: torch.Tensor) -> np.ndarray:
    batch_tensor = batch_input_tensor.to(device)
    with torch.no_grad():
        features = _cnn_extractor(batch_tensor)
        features = torch.flatten(features, 1)
        embeddings = features.cpu().numpy()
    return embeddings

def new_cnn_extractor() -> Sequential:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    cnn_extractor = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)))
    return cnn_extractor

