import torch
import cv2
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
from skimage.color import gray2rgb
from PIL import Image
import sys

# -*- coding: utf-8 -*-
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication,QLabel,QLineEdit,QGridLayout
from PyQt5.QtGui import QIcon
from PyQt5 import QtCore, QtWidgets

import sys
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Ui_MainWindow(object):

    def setupUi(self, MainWindow):

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(848, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")


        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(27,100,800,300))
        self.textEdit.setObjectName("text")



        self.file = QtWidgets.QPushButton(self.centralwidget)
        self.file.setGeometry(QtCore.QRect(57, 460, 175, 28))
        self.file.setObjectName("file1")
        self.file.setStyleSheet("background-color:rgb(111,180,219)")
        self.file.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"  
            "QPushButton:hover{color:green}"  
            "QPushButton{border-radius:6px}" 
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"
        )


        self.fileT = QtWidgets.QPushButton(self.centralwidget)
        self.fileT.setGeometry(QtCore.QRect(300, 460, 480, 28))
        self.fileT.setObjectName("file2")
        self.fileT.setStyleSheet("background-color:rgb(111,180,219)")
        self.fileT.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"
            "QPushButton:hover{color:green}"
            "QPushButton{border-radius:6px}"
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"
        )
        self.st = QtWidgets.QPushButton(self.centralwidget)
        self.st.setGeometry(QtCore.QRect(57, 520, 175, 28))
        self.st.setObjectName("file3")
        self.st.setStyleSheet("background-color:rgb(111,180,219)")
        self.st.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"
            "QPushButton:hover{color:green}" 
            "QPushButton{border-radius:6px}" 
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"
        )
        self.sst = QtWidgets.QPushButton(self.centralwidget)
        self.sst.setGeometry(QtCore.QRect(300, 520, 175, 28))
        self.sst.setObjectName("file3")
        self.sst.setStyleSheet("background-color:rgb(111,180,219)")
        self.sst.setStyleSheet(
            "QPushButton{background-color:rgb(111,180,219)}"
            "QPushButton:hover{color:green}" 
            "QPushButton{border-radius:6px}" 
            "QPushButton:pressed{background-color:rgb(180,180,180);border: None;}"
        )
        self.p = QtWidgets.QLabel(self.centralwidget)
        self.p.setGeometry(QtCore.QRect(300, 60, 480, 28))


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 848, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslatUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



        self.file.clicked.connect(self.msg)
        self.st.clicked.connect(self.dim1)
        self.sst.clicked.connect(self.dim2)




    def retranslatUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "敏感图片分类"))
        self.file.setText(_translate("MainWindow", "选择文件"))
        self.fileT.setText(_translate("MainWindow", ""))
        self.st.setText(_translate("MainWindow","色情检测"))
        self.sst.setText(_translate("MainWindow","血腥检测"))
        self.textEdit.setText(_translate("MainWindow",""))



    def msg(self, Filepath):
        m = QtWidgets.QFileDialog.getExistingDirectory(None, "选取文件夹", "C:/")  # 起始路径
        self.fileT.setText(m)
        self.p.setText("预测结果为：")


    def dim1(self,):
        s = self.fileT.text()

        image_path = s + r'/'
        path = './paramsm6-5.pkl'

        cnn = ResNet34()
        cnn.load_state_dict(torch.load(path))
        cnn.to(device)
        cnn.eval()
        trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        classes = ['drawing', 'sexy', 'porn']
        self.textEdit.clear()
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

            self.textEdit.insertPlainText("预测：" + result + "     图片：" + files + "\n")
    def dim2(self,):
        s = self.fileT.text()

        image_path = s + r'/'
        path = './params-b8-5.pkl'

        cnn = ResNet34()
        cnn.load_state_dict(torch.load(path))
        cnn.to(device)
        cnn.eval()
        trans = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        classes = ['drawing', 'bloody']
        self.textEdit.clear()
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

            self.textEdit.insertPlainText("预测：" + result + "     图片：" + files + "\n")







if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    mainWindow = QtWidgets.QMainWindow()

    ui = Ui_MainWindow()

    ui.setupUi(mainWindow)

    mainWindow.show()

    sys.exit(app.exec_())
