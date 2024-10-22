import cv2
import pytesseract
import numpy as np

# test
# 读取图片
image_path = './111.png'
img = cv2.imread(image_path)

# 将图片转为灰度模式
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 设定一个阈值，区分浅色和深色
_, thresh_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 使用pytesseract进行字符识别，配置只识别数字
custom_config = r'--oem 3 --psm 6 outputbase digits'
detected_text = pytesseract.image_to_string(thresh_img, config=custom_config)

# 获取每个字符的位置信息
h, w, _ = img.shape
boxes = pytesseract.image_to_boxes(thresh_img, config=custom_config)

# 初始化一个空网格
grid = [['H' for _ in range(w // 16)] for _ in range(h // 16)]

# 根据 boxes 替换数字位置
for box in boxes.splitlines():
    b = box.split(' ')
    char, x, y, x2, y2 = b[0], int(b[1]), int(b[2]), int(b[3]), int(b[4])
    grid_pos_x = x // 16
    grid_pos_y = (h - y) // 16
    grid[grid_pos_y][grid_pos_x] = char

# 将深色部分转为 '0'
for i in range(h // 16):
    for j in range(w // 16):
        if thresh_img[i * 16, j * 16] < 127:  # 深色部分
            grid[i][j] = '0'

# 打印输出
for row in grid:
    print(''.join(row))
