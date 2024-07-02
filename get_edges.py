import os
import torch
import random
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt
from helper import PathException
from model import pidiNet
from PIL import Image
import cv2
from skimage import morphology,draw
import torchvision.transforms as transforms
plt.rc("font",family='Microsoft YaHei')

# 初始化网络为None
netNetwork = None
# 模型目录
modeldir = r"E:\project_crop\control_nematode\control_dege\pidinet\load_params"

def apply_pidinet(input_image, is_safe=False, apply_fliter=False):
    try:
        global netNetwork
        # 如果网络为空，则加载模型
        if netNetwork is None:
            # 模型路径
            modelpath = os.path.join(modeldir, "table5_pidinet.pth")

            if not os.path.exists(modelpath):
                raise PathException(modelpath)

            # 创建网络实例
            netNetwork = pidiNet()
            # netNetwork = torch.nn.DataParallel(netNetwork).cuda()
            # 加载模型状态字典
            ckp = torch.load(modelpath, map_location='cpu')['state_dict']
            # 更新网络状态字典
            netNetwork.load_state_dict({k.replace('module.', ''): v for k, v in ckp.items()})
            # model.load_state_dict(checkpoint['state_dict'])

        netNetwork.eval()


        # 确保输入图像维度为3
        assert input_image.ndim == 3
        # 将图像通道顺序从RGB转为BGR
        input_image = input_image[:, :, ::-1].copy()
        # 在不计算梯度的情况下运行网络
        with torch.no_grad():
            # 将输入图像转为张量并归一化
            image_pidi = torch.from_numpy(input_image).float()
            image_pidi = image_pidi / 255.0
            # 调整张量的维度顺序
            image_pidi = rearrange(image_pidi, 'h w c -> 1 c h w')
            # 通过网络获取边缘信息
            edge = netNetwork(image_pidi)[-1]
            # 将边缘信息转为numpy数组
            edge = edge.cpu().numpy()
            # 如果需要应用过滤器，则对边缘信息进行处理
            if apply_fliter:
                edge = edge > 0.5
            # 如果需要安全处理，则对边缘信息进行处理
            if is_safe:
                edge = safe_step(edge)
            # 将边缘信息归一化并裁剪到0-255范围内，然后转为整数类型
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
        # 返回处理后的边缘信息
        return edge[0][0]

    except PathException as e:
        print(e)

def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y

# 细化算法
def get_thinning_edge(edge):
    thinned = cv2.ximgproc.thinning(edge)
    return thinned

# canny
def get_canny_edge(img):
    # 转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Canny边缘检测
    edges = cv2.Canny(blur, 100, 200)
    return edges

def delete_small_edge(img, threthod=100):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 设置最小边界长度阈值
    min_length = threthod
    # 去除小块区域噪声
    for contour in contours:
        length = cv2.arcLength(contour, True)
        if length < min_length:
            cv2.drawContours(img, [contour], -1, 0, thickness=cv2.FILLED)
    return img

def delete_small_area(img, threthod=100):
    # 连通组件分析
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img)
    # 设置最小面积阈值
    min_area = threthod
    # 去除小块区域噪声
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            labels[labels == label] = 0

    # 将去除小块区域的结果保存为二值图像
    result = (labels > 0).astype(np.uint8) * 255
    return result


def random_walk(target_path, num_images=5):
    image_files = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return random.sample(image_files, num_images)

if __name__ == '__main__':

    filePath = r'E:\研究生\Few-shot Nematode Image Generation based on Diffusion Models and Morphological Constraints\Nema-SD\绘图\边缘\Xh\x.jpg'
    path_1 = r'E:\研究生\Few-shot Nematode Image Generation based on Diffusion Models and Morphological Constraints\Nema-SD\绘图\边缘\Xh'
    savePath = os.path.join(path_1, 'edge1.png')
    savePath_1 = os.path.join(path_1, 'edge2.png')
    savePath_2 = os.path.join(path_1, 'edge3.png')
    # files = random_walk(filePath, 1)
    # for img_path in files:
    for i in range(1):
        # img = np.array(Image.open(img_path))
        img = np.array(Image.open(filePath))

        pidi_edges = apply_pidinet(img)

        plt.figure()
        plt.subplot(2, 4, 1)
        plt.imshow(pidi_edges, cmap='gray')
        plt.title("image")

        # thin_edge = delete_small_area(pidi_edges, 300)
        thin_edge = get_thinning_edge(pidi_edges)
        plt.subplot(2, 4, 2)
        plt.imshow(thin_edge, cmap='gray')
        plt.title("pidiNet")
        thin_edge_s = Image.fromarray(thin_edge)
        thin_edge_s.save(savePath, format("png"))


        # edges = get_thinner_edge(edges)
        # thin_edge = get_thinning_edge(pidi_edges)
        thin_edge_1 = delete_small_area(thin_edge, 300)
        plt.subplot(2, 4, 3)
        plt.imshow(thin_edge_1, cmap='gray')
        plt.title("Thinning_pidiNet")

        thin_edge_2 = delete_small_edge(thin_edge, 300)
        plt.subplot(2, 4, 4)
        plt.imshow(thin_edge_2, cmap='gray')
        plt.title("Thinning_pidiNet")

        canny_edge = get_canny_edge(img)
        print(type(canny_edge))
        plt.subplot(2, 4, 5)
        plt.imshow(canny_edge, cmap='gray')
        plt.title("Canny")

        delete_edge = delete_small_edge(canny_edge, 300)
        plt.subplot(2, 4, 6)
        plt.imshow(delete_edge, cmap='gray')
        plt.title("Canny_delete_edge")

        delete_area = delete_small_area(canny_edge, 300)
        plt.subplot(2, 4, 7)
        plt.imshow(delete_area, cmap='gray')
        plt.title("Canny_delete_area")

        plt.show()


        pidi_edges = Image.fromarray(thin_edge_2)
        pidi_edges.save(savePath_1, format("png"))
        thin_edge_1 = Image.fromarray(thin_edge_1)
        thin_edge_1.save(savePath_2, format("png"))
    # path = r'E:\project_crop\control_nematode\dataset\resize\png\00002.png'
    # im = np.array(Image.open(path))
    # edges = apply_pidinet(im)
    # get_thinner_edge(edges)

    # plt.figure()
    # plt.imshow(edges, cmap='gray')
    # plt.title("边缘图")
    # plt.show()

