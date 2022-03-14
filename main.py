import skimage
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
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



# step1: 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):

    def __init__(self, root_dir, names_file, transform=None):
        self.root_dir = root_dir
        self.names_file = names_file
        self.transform = transform
        self.size = 0
        self.names_list = []

        if not os.path.isfile(self.names_file):
            print(self.names_file + 'does not exist!')
        file = open(self.names_file)
        for f in file:
            self.names_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = self.root_dir + self.names_list[idx].split(' ')[0]
        print(image_path)
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        image = io.imread(image_path)   # use skitimage
        if len(image.shape) ==2:
            image = gray2rgb(image)
        if len(image.shape) ==4:
            image = rgba2rgb(rgba=image, background=(1,1,1))
        label = int(self.names_list[idx].split(' ')[1])
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

# # 变换Resize
class Resize(object):

    def __init__(self, output_size: tuple):
        self.output_size = output_size

    def __call__(self, sample):
        # 图像
        image = sample['image']
        # 使用skitimage.transform对图像进行缩放

        image_new = transform.resize(image, self.output_size)
        return {'image': image_new, 'label': sample['label']}
'''
        if len(image_new.shape) == 2:
            print('dim=', 2)
            image_new = gray2rgb(image_new)
        if len(image_new.shape) == 3:
            print('dim=', 3)
        if len(image_new.shape) == 4:
            print('dim=', 4)
            image_new = skimage.color.rgba2rgb(image_new, background=(1, 1, 1))
            if len(image_new.shape) == 3:
                print('trans dim complete!')
'''

# # 变换ToTensor
class ToTensor(object):

    def __call__(self, sample):
        image = sample['image']
        image_new = np.transpose(image, (2, 0, 1))
        image_new = torch.from_numpy(image_new)
        return {'image': image_new,'label': sample['label']}

# 对原始的训练数据集进行变换
def train_data():

    train_dataset = MyDataset(root_dir='./data2',names_file='./label.txt', transform=transforms.Compose([Resize((32,32)), ToTensor()]))
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True, num_workers=0, drop_last=True)
    return trainloader

def train_save():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = ResNet34()
    path = 'D:\\PythonProjects\\for learn\\params-b8-4.pkl'
    net.load_state_dict(torch.load(path))

    net.to(device)
    trainloader = train_data()
    # print(trainloader)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.009), eps=1e-08, weight_decay=0.001, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    for epoch in range(5):
        running_loss = 0.0
        for i,data in enumerate(trainloader,0):
            inputs = data['image']
            labels = data['label']
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            inputs = inputs.to(device)
            labels = labels.to(device)

            print('开始训练图片:', i + 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()  # tensor.item()
            # if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))  # 然后再除以200，就得到这两百次的平均损失值
            running_loss = 0.0
    print('训练完成')
    torch.save(net.state_dict(), 'params-b8-5.pkl')


if __name__ == "__main__":
    train_save()



# if __name__ == "__main__":
#     train_dataset = MyDataset(root_dir='./data', names_file='./label.txt',
#                               transform=transforms.Compose([Resize((224, 224, 3)), ToTensor()]))
#     plt.figure()
#     for va in train_dataset:
#         print(va)
#
#     for (cnt, i) in enumerate(train_dataset):
#         image = i['image']
#         label = i['label']
#        # print(image)
#         print(label)
