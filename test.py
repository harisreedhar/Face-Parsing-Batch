import os
import cv2
import torch
import numpy as np
from face_parsing import FACEPARSER


# initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
fp = FACEPARSER(model_path="pretrained_model/79999_iter.pth", device=device)


# get images
directory = "test/faces"
images = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        images.append(cv2.imread(image_path))


# get masks
masks = fp.get_mask(images, classes=[1, 2, 3, 4, 5, 10, 11, 12, 13], batch_size=32)


# write
for i, (image, mask) in enumerate(zip(images, masks)):
    mask = mask.astype('uint8') * 255
    mask = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (image.shape[0], image.shape[0]))
    output_stack = np.hstack((image, mask))
    cv2.imwrite(f"test/result/result_{i}.jpg", output_stack)
