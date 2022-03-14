import cv2
import skimage
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
import PIL.Image
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from net import ResNet34
from skimage import io
from skimage import transform
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from skimage.color import gray2rgb,rgba2rgb

class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        image_new = np.transpose(image, (2, 0, 1))
        image_new = torch.from_numpy(image_new)
        return {'image': image_new,'label': sample['label']}

class Resize(object):

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        # 图像
        image = sample['image']
        # 使用skitimage.transform对图像进行缩放

        image_new = transform.resize(image, self.output_size)
        return {'image': image_new, 'label': sample['label']}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def tmod():
    x = 0
    classes = ['drawing','sexy','porn']
    image_path = 'D:\\PythonProjects\\for learn\\data2\\drawing2\\'
    path = 'D:\\PythonProjects\\for learn\\paramsm6-5.pkl'
    trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    cnn = ResNet34()
    cnn.load_state_dict(torch.load(path))
    cnn.to(device)
    cnn.eval()
    for files in os.listdir(image_path):

        image_paths = image_path + files
        image = PIL.Image.open(image_paths)
        tsfmd_image = trans(image).unsqueeze(0)
        tsfmd_image = tsfmd_image.to(device)
        outputs = cnn(tsfmd_image)
        _, indices = torch.max(outputs, 1)
        percentage = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        perc = percentage[int(indices)].item()
        result = classes[indices]
        if result =='drawing':
            x += 1
        print('predicted:', result)


if __name__ == "__main__":
    tmod()