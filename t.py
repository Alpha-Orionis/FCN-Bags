#######t.py
# 步骤加载模型
# 设置transform或者cv2将图像转化成想输入的大小
# 传入模型输出显示或保存
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import os
import cv2
import matplotlib.pyplot as plt
from BagData import test_dataloader, train_dataloader
from FCN import  FCNs, VGGNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('checkpoints/fcn_model_95.pt', map_location='cpu')# 加载模型
model = model.to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

if __name__ == '__main__':
    img_name = r'88.jpg'  # 预测的图片
    imgA = cv2.imread(img_name)
    imgA = cv2.resize(imgA, (160, 160))
    imgC = imgA
    imgC = np.array(imgC)
    imgA = transform(imgA)
    imgA = imgA.to(device)
    imgA = imgA.unsqueeze(0)  # 在其第一个维度前再增加一个维度，期为读书为1；保证能顺利放进网络，该1的本质是batchsize
    output = model(imgA)
    output = torch.sigmoid(output)

    output_np = output.cpu().detach().numpy().copy()  # output_np.shape = (4, 2, 160, 160)
    print(output_np.shape)  # (1, 2, 160, 160)
    output_np = np.argmin(output_np, axis=1)
    print(output_np.shape)  # (1,160, 160)

    plt.subplot(1, 2, 1)
    # plt.imshow(np.squeeze(bag_msk_np[0, ...]), 'gray')
    # plt.subplot(1, 2, 2)
    print(output_np.shape)
    #plt.imshow(np.squeeze(output_np[0, ...]), 'gray')  # 最后利用np.squeeze()将维度转化为2维，并以灰度图形式读取

    #im = plt.pause(3)
    we = np.squeeze(output_np[0, ...])
    for y in range(160):
        for x in range(160):
            if (we[x, y] == 1 ):
                imgC[x, y, 0] = 255
                imgC[x, y, 1] = 255
                imgC[x, y, 2] = 255

    plt.imshow(imgC)






