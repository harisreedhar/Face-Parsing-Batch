import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms
from . model import BiSeNet


transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def transform_images(imgs):
    tensor_images = torch.stack(
        [transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in imgs], dim=0
    )
    return tensor_images


class FACEPARSER:
    def __init__(self, model_path="79999_iter.pth", device="cpu"):
        self.device = device
        self.net = BiSeNet(19)
        self.net.to(self.device)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()


    def get_mask(self, imgs, classes=[1, 2, 3, 4, 5, 10, 11, 12, 13], batch_size=8):
        mask_list = []

        for i in tqdm(range(0, len(imgs), batch_size), total=len(imgs) // batch_size, desc="Face-parsing"):
            batch_imgs = imgs[i:i + batch_size]

            tensor_images = transform_images(batch_imgs).to(self.device)
            with torch.no_grad():
                out = self.net(tensor_images)[0]

            parsing = out.argmax(dim=1).detach().cpu().numpy()
            batch_masks = np.isin(parsing, classes)
            mask_list.extend(batch_masks)

        # TODO: batch resize to original resolution

        return mask_list
