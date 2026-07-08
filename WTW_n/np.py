import numpy as np
from PIL import Image

# 读取图片
image = Image.open(r"D:\deeplearning\WTW\seg_ak\TWB_OD_B_img001_pred3.png")  # 替换为你的图片文件名

# 将图片转换为numpy数组
image_array = np.array(image)

# 打印图片的像素值
print(image_array)

image_array = np.array(image)
print(image_array.shape)

# # 打印所有像素值
# for row in image_array:
#     for pixel in row:
#         print(pixel)