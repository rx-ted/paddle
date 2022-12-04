'''
Author: rx-ted
Date: 2022-12-04 12:30:31
LastEditors: rx-ted
LastEditTime: 2022-12-04 17:10:56
'''
import random
import cv2
import paddle
from paddle.io import DataLoader, Dataset
import paddle.optimizer as O
import paddle.nn.functional as F
import paddle.nn.layer as L
import paddle.nn as nn
import os
import matplotlib.pyplot as plt
import numpy as np
import zipfile
food_zip = 'food-11.zip'
"""

Bread, Dairy product, Dessert, Egg, Fried food, Meat, Noodles/Pasta, Rice, Seafood, Soup, and Vegetable/Fruit. 
（面包，乳制品，甜点，鸡蛋，油炸食品，肉类，面条/意大利面，米饭，海鲜，汤，蔬菜/水果） 
Training set: 9866张 Validation set: 3430张 Testing set: 3347张


数据格式 下载 zip 档后解压缩会有三个资料夹，分别为training、validation 以及 testing training 以及 validation 中的照片名称格式为 [类别]_[编号].jpg，例如 3_100.jpg 即为类别 3 的照片（编号不重要）
"""
label_dict = {
    0: ['Bread', '面包'], 1: ['Dairy product', '乳制品'], 2: ['Dessert', "甜点"],
    3: ['Egg', '鸡蛋'], 4: [' Fried food', '油炸食品'], 5: ['Meat', '肉类'],
    6: ['Noodles/Pasta', '面条/意大利面'], 7: ['Rice', '米饭'], 8: ['Seafood', '海鲜'],
    9: ['Soup', '汤'], 10: ['Vegetable/Fruit', '蔬菜/水果']
}


def unzip(file, output):
    f = zipfile.ZipFile(file, 'r')
    f.extractall(output)
    f.close()
# unzip(food_zip,None)


device = paddle.device.set_device('GPU')
print(device)
# image size is different.


def preprocess(img, mode='train'):
    img = cv2.resize(img, (128, 128))
    # 在训练集中随机对数据进行flip操作
    if mode == 'train':
        if random.randint(0, 1):  # 随机进行预处理
            img = cv2.flip(img, random.randint(-1, 1))  # flip操作模式随机选择
    # 转换为numpy数组
    img = np.array(img).astype('float32')
    # 将数据范围改为0-1
    img = img / 255.
    # 最后更改数组的shape，使其符合CNN输入要求
    return img.transpose((2, 0, 1))


class FoodDataSet(paddle.io.Dataset):
    def __init__(self, data_dir, mode):
        # 获取文件夹下数据名称列表
        self.filenames = os.listdir(data_dir)
        self.data_dir = data_dir
        self.mode = mode

    def __getitem__(self, index):
        file_name = self.data_dir + self.filenames[index]
        # 读取数据
        img = cv2.imread(file_name)
        # 预处理
        img = preprocess(img, mode=self.mode)
        # 获得标签
        label = int(self.filenames[index].split('_')[0])
        return img, label

    def __len__(self):
        return len(self.filenames)


train_dataset = FoodDataSet('food-11/training/', 'train')
train_loader = DataLoader(train_dataset, places=device,
                          batch_size=64, shuffle=True)


eval_dataset = FoodDataSet('food-11/validation/', 'validation')
eval_loader = DataLoader(eval_dataset, places=device,
                         batch_size=64, shuffle=True)


class LeNet(nn.Layer):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv0 = nn.Conv2D(
            in_channels=3, out_channels=10, kernel_size=5, padding="SAME", stride=1)
        self.pool0 = nn.MaxPool2D(
            kernel_size=2, stride=2)  # 128 * 128 -> 64 * 64

        self.conv1 = nn.Conv2D(
            in_channels=10, out_channels=20, kernel_size=5, padding="SAME", stride=1)
        self.pool1 = nn.MaxPool2D(
            kernel_size=2, stride=2)  # 64 * 64 -> 32 * 32

        self.conv2 = nn.Conv2D(
            in_channels=20, out_channels=50, kernel_size=5, padding="SAME", stride=1)
        self.pool2 = nn.MaxPool2D(
            kernel_size=2, stride=2)  # 32 * 32 -> 16 * 16

        self.fc1 = nn.Linear(in_features=12800, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=11)

    def forward(self, x):
        x = self.conv0(x)
        x = nn.functional.leaky_relu(x)
        x = self.pool0(x)

        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.pool2(x)

        x = paddle.reshape(x, [x.shape[0], -1])

        x = self.fc1(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc2(x)
        x = nn.functional.leaky_relu(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x)
        return x


# network = LeNet()

# # paddle.summary(network, (1, 3, 128, 128))

# model = paddle.Model(network)

# model.prepare(O.Adam(learning_rate=0.0001, parameters=model.parameters()),
#               nn.CrossEntropyLoss(),
#               paddle.metric.Accuracy())

# visualdl = paddle.callbacks.VisualDL(log_dir='visualdl_log')

# # 启动模型全流程训练
# model.fit(train_loader,  # 训练数据集
#           eval_loader,   # 评估数据集
#           epochs=10,       # 训练的总轮次
#           batch_size=128,  # 训练使用的批大小
#           verbose=1,      # 日志展示形式
#           callbacks=[visualdl])  # 设置可视化

# model.save('model/LeNet')

def predict(file):

    model_state_dict = paddle.load('model\LeNet.pdparams')
    model = LeNet()
    model.set_state_dict(model_state_dict)
    model.eval()

    # text_dir = 'food-11/testing/'
    # test_filename = os.listdir(text_dir)
    # file = text_dir + test_filename[1]
    
    img = cv2.imread(file)

    plt.imshow(img[:, :, ::-1])
    plt.show()

    img = preprocess(img, mode='test')
    res = model(paddle.to_tensor(img[np.newaxis, :, :, :]))
    g = preprocess(img, mode='test')
    res = model(paddle.to_tensor(img[np.newaxis, :, :, :]))
    idx = np.argmax(res.numpy())
    print(label_dict[idx])


for i in range(14):
    file = 'test/'+str(i)+'.jpg'
    predict(file)
    

