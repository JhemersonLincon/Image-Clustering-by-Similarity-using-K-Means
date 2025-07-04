import cv2 as cv
import numpy as np
from torchvision.models import resnet18
from torch.nn import Sequential
from torch import no_grad
# === Funções para uma imagem ===

def extract_canny_features(image):  
    img_blur = cv.GaussianBlur(image, (3, 3), 0) 
    edges = cv.Canny(image=img_blur, threshold1=100, threshold2=200) 
    return edges

def extract_canny_features_from_dataset(images_path):
    images_area = []
    for image_path in images_path:
        image = cv.imread(image_path)
        edges = extract_canny_features(image)
        images_area.append((edges / 255).sum())
    return np.array(images_area, dtype=np.float32).reshape(-1, 1)

def extract_resnet_features(image):
    model = resnet18(weights="DEFAULT")
    
    conv1 = Sequential(model.conv1) 
    feature_extractor = Sequential(*list(model.children())[:-1])
    with no_grad():
        image_features = conv1(image)
        features = feature_extractor(image)

    return image_features, features

