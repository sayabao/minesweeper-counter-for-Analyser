import cv2
import numpy as np

# 加载图像
image_path = './111.png'
image = cv2.imread(image_path)

# 定义颜色范围 (使用 HSV 格式)
color_ranges = {
    '0': ((100, 53, 60), (110, 67, 70)), # 扩大灰色范围
    '1': ((100, 69, 85), (110, 79, 95)),  # 扩大颜色范围
    '2': ((80, 160, 80), (105, 190, 105)),   # 扩大绿色范围
}

# 初始化矩阵
num_rows, num_cols = 16, 30
matrix = [['H' for _ in range(num_cols)] for _ in range(num_rows)]

# 将图像转换为 HSV 颜色空间
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义网格单元的大小
cell_height = hsv_image.shape[0] // num_rows
cell_width = hsv_image.shape[1] // num_cols

# 遍历每个网格单元并进行颜色检测
for row in range(num_rows):
    for col in range(num_cols):
        # 计算每个单元格的坐标
        x_start = col * cell_width
        y_start = row * cell_height
        x_end = x_start + cell_width
        y_end = y_start + cell_height

        # 截取每个单元格区域
        cell = hsv_image[y_start:y_end, x_start:x_end]

        # 计算平均颜色值
        avg_color = cv2.mean(cell)[:3]  # 提取平均颜色的 HSV 值

        # 初始化标记为隐藏 ('H')
        detected_value = 'H'

        # 根据平均颜色值决定矩阵的值
        for num, (lower, upper) in color_ranges.items():
            h_check = lower[0] <= avg_color[0] <= upper[0]
            s_check = lower[1] <= avg_color[1] <= upper[1]
            v_check = lower[2] <= avg_color[2] <= upper[2]
            
            #print(f"Checking {num}: H: {h_check}, S: {s_check}, V: {v_check}")

            if h_check and s_check and v_check:
                detected_value = num
                break

        # 将检测到的值和平均颜色填入矩阵
        matrix[row][col] = (detected_value, avg_color)  # 存储数字和颜色值

# 打印矩阵
for row in matrix:
    print(row)
