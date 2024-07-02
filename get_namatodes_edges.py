import os
import torch
import random
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
import pidinet.get_edges as ge
from PIL import Image
import torchvision.transforms.functional as TF
import cv2
from skimage import morphology,draw
import torchvision.transforms as transforms
plt.rc("font",family='Microsoft YaHei')

nema_list = ['Pr', 'Me', 'Xh', 'Tr', 'Lo']

for nema in nema_list:
    file_path = 'E:\\线虫数据\\control_nematode_DS\\SD_set\\' + nema + '\\原图_1024'
    target_path = 'E:\\线虫数据\\control_nematode_DS\\SD_set\\' + nema + '\\轮廓图'
    files = os.listdir(file_path)
    for image in files:
        img = np.array(Image.open(os.path.join(file_path, image)))

        pidi_edges = ge.apply_pidinet(img)
        thin_edge = ge.get_thinning_edge(pidi_edges)
        edges = ge.delete_small_area(thin_edge, 300)
        edges = TF.resize(Image.fromarray(edges), (512, 512), interpolation=Image.BICUBIC)

        img_path = os.path.join(target_path, image.split('.')[0] + '.png')
        edges.save(img_path)
        print(img_path, '保存成功！')
