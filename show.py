# 展示MNIST数据集里的图片和对应标签
# t10k开头的是测试集
# train开头的是训练集
# idx3是图像数据
# idx1是标签数据
# .gz是对应文件的压缩包，不用管

import struct
import numpy as np
import matplotlib.pyplot as plt
import random

def load_images(filename):
    with open(filename, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images

def load_labels(filename):
    with open(filename, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))  
        labels = np.fromfile(f, dtype=np.uint8) 
    return labels

def show_image(images, labels, index):
    plt.imshow(images[index], cmap='gray') 
    plt.title(f"Label: {labels[index]}") 
    plt.show()

train_image_file = 'data\\MNIST\\raw\\train-images-idx3-ubyte'
train_label_file = 'data\\MNIST\\raw\\train-labels-idx1-ubyte'

train_images = load_images(train_image_file)
train_labels = load_labels(train_label_file)

index = random.randint(0, train_labels.size-1)
print(f'第{index}张图片')
show_image(train_images, train_labels, index)
