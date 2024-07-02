import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import time

try:
    import torch  # Install PyTorch first: https://pytorch.org/get-started/locally/
    from deformation import (
        mls_affine_deformation as mls_affine_deformation_pt,
        mls_similarity_deformation as mls_similarity_deformation_pt,
        mls_rigid_deformation as mls_rigid_deformation_pt,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
except ImportError as e:
    print(e)

img_path = r'E:\dataSet\1225 轮廓图\Me.png'

def random_walk(target_path, num_images=5):
    image_files = []
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))
    return random.sample(image_files, num_images)

def delete_small_area(img, threthod=100):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def rotation_points(points):
    alpha = 0
    flag = True
    count = 50
    while alpha**2 < 1 or flag:
        count -= 1
        # alpha = random.randint(-25, 25)
        alpha = random.choice([25, -25])
        flag = False
        # 计算旋转矩阵
        angle = np.radians(alpha)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        # 将第一个点作为旋转中心
        center_point = points[0]
        # 将所有点相对于旋转中心平移到原点
        translated_points = points - center_point
        # 将所有点绕旋转中心旋转
        rotated_points = np.dot(translated_points, rotation_matrix.T)
        # 将旋转后的点平移回原来的位置
        final_points = rotated_points + center_point
        for pot in final_points:
            i, j = pot
            if i < 0 or i >500 or j < 0 or j > 500:
                flag = True
                break
        if count == 0:
            break
    # 绘制旋转前后的图像
    # plt.scatter(points[:, 0], points[:, 1], c='r', label='Original Points')
    # plt.scatter(final_points[:, 0], final_points[:, 1], c='b', label='Rotated Points')
    # plt.plot(points[:, 0], points[:, 1], 'r-', alpha=0.3)
    # plt.plot(final_points[:, 0], final_points[:, 1], 'b-', alpha=0.3)
    # plt.legend()
    # plt.axis('equal')
    # plt.show()

    return final_points, count

# nema_list = ['Pr', 'Me', 'Xh', 'Tr', 'Lo']
nema_list = ['Me']
for nema in nema_list:
    file_path = 'E:\\线虫数据\\control_nematode_DS\\SD_set\\' + nema + '\\轮廓图'
    # file_path = 'D:\\Desktop\\desktop\\Nema-SD\\绘图\\轮廓约束'
    target_path = 'E:\\线虫数据\\control_nematode_DS\\SD_set\\' + nema + '\\生成约束'
    # files = random_walk(file_path, 5)
    # for image in files:
    files = os.listdir(file_path)
    idx = 1
    while idx < 1000:
        file = random.choice(files)
        image = os.path.join(file_path, file)
        print(image)

        edges = cv2.imdecode(np.fromfile(image,dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        # print(edges)
        # 获取边缘点的坐标
        points = np.where(edges != 0)
        points = np.column_stack((points[1], points[0]))
        # 随机选择256个边缘点
        np.random.seed(42)
        random_indices = np.random.choice(len(points), size=256, replace=False)
        random_edge_points = points[random_indices]
        # print(random_edge_points)

        coefficients = np.polyfit(random_edge_points[:, 0], random_edge_points[:, 1], deg=1)
        line = np.poly1d(coefficients)

        print(line)
        # plt.figure()
        # plt.imshow(edges, cmap='gray')
        # plt.scatter(random_edge_points[:, 0], random_edge_points[:, 1], c='r', s=5)
        # plt.plot(random_edge_points[:, 0], line(random_edge_points[:, 0]), c='g', label='Fitted Line')
        # plt.show()

        a = coefficients[0]
        b = coefficients[1]
        x_min = -b / a
        x_max = (500-b) / a
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        min_len = random_edge_points.min() if random_edge_points.min() > x_min else x_min
        max_len = random_edge_points.max() if random_edge_points.max() < x_max else x_max
        x_line = np.linspace(min_len, max_len, 8)
        y_line = a * x_line + b
        select_points = np.column_stack((x_line, y_line))
        # print(select_points)
        final_points, count = rotation_points(select_points)
        if count == 0:
            continue
        # print(final_points)

        p = torch.from_numpy(np.array(np.flip(select_points, axis=1))).to(device)
        q = torch.from_numpy(np.array(np.flip(final_points, axis=1))).to(device)
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        image = torch.from_numpy(np.array(img)).to(device)
        print(image.shape)

        height, width, _ = image.shape
        gridX = torch.arange(width, dtype=torch.int16).to(device)
        gridY = torch.arange(height, dtype=torch.int16).to(device)
        vy, vx = torch.meshgrid(gridX, gridY)

        vy, vx = vy.transpose(0, 1), vx.transpose(0, 1)

        similar = mls_similarity_deformation_pt(vy, vx, p, q, alpha=1)
        aug2 = torch.ones_like(image).to(device)
        aug2[vx.long(), vy.long()] = image[tuple(similar)]

        print(type(aug2.cpu().numpy()))
        result = delete_small_area(aug2.cpu().numpy(), 400)

        angles = np.random.randint(0, 4)*90
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 水平随机翻转，概率为0.5
            transforms.RandomRotation(degrees=[angles, angles]),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
        ])
        result = transform(Image.fromarray(result))
        img_name = nema + "_{:0>3}".format(idx) + '.png'
        save_path = os.path.join(target_path, img_name)
        # result.save(save_path, format("png"))
        print('save: ', img_name, '.png')
        idx += 1


        # 显示原图和点集
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6))
        ax1.imshow(edges, cmap='gray')
        ax1.set_title('Original Image')
        ax2.imshow(edges, cmap='gray')

        savePath = r'D:\Desktop\desktop\Nema-SD\绘图\轮廓约束\edge4.png'
        save_edge = Image.fromarray(edges)
        save_edge.save(savePath, format("png"))

        ax2.scatter(random_edge_points[:, 0], random_edge_points[:, 1], c='r', s=5)
        ax2.set_title('Edge Points')
        ax2.plot(random_edge_points[:, 0], line(random_edge_points[:, 0]), c='g', label='Fitted Line')
        ax2.scatter(x_line, y_line, color='b', label='Points on Fitted Line')
        ax2.scatter(final_points[:, 0], final_points[:, 1], color='purple', label='Points on Fitted Line')
        ax3.imshow(aug2.cpu().numpy())
        ax3.set_title("Similarity Deformation")

        savePath = r'D:\Desktop\desktop\Nema-SD\绘图\轮廓约束\edge5.png'
        save_edge = Image.fromarray(aug2.cpu().numpy())
        save_edge.save(savePath, format("png"))

        ax4.imshow(result, cmap='gray')
        ax4.set_title('Result')

        savePath = r'D:\Desktop\desktop\Nema-SD\绘图\轮廓约束\edge6.png'
        result.save(savePath, format("png"))

        plt.show()












