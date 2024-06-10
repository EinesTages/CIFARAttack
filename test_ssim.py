import os
from skimage import io
from skimage.metrics import structural_similarity
import cv2


def calculate_ssim(folder1, folder2):
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        raise ValueError("One or both of the folders do not exist.")

    images1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.png')]
    images2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.png')]

    if len(images1) != len(images2):
        raise ValueError("The number of images in the two folders must be the same.")

    ssim_values = []

    # 计算每对图片的SSIM值
    for i in range(len(images1)):
        img1 = cv2.imread(images1[i])
        img2 = cv2.imread(images2[i])
        ssim_value = structural_similarity(img1, img2, multichannel=True,channel_axis=2)
        ssim_values.append(ssim_value)

    # 计算最大值、最小值和均值
    max_ssim = max(ssim_values)
    min_ssim = min(ssim_values)
    mean_ssim = sum(ssim_values) / len(ssim_values)

    return max_ssim, min_ssim, mean_ssim


# 替换以下路径为你的文件夹路径
folder_path1 = './data/cifar10_clean_500/images'
folder_path2 = './submit'

# 调用函数并打印结果
max_ssim, min_ssim, mean_ssim = calculate_ssim(folder_path1, folder_path2)
print(f"Max SSIM: {max_ssim}")
print(f"Min SSIM: {min_ssim}")
print(f"Mean SSIM: {mean_ssim}")
