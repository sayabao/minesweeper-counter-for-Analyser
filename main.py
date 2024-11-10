import cv2
import numpy as np

# 第一步：读取并裁剪图像
image = cv2.imread("map.png")
if image is None:
    raise ValueError("Image not found or cannot be loaded.")

# 转为灰度并进行边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 20, 100)
edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
edges = cv2.erode(edges, None)

# 寻找四边形轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
found_corners = False
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(approx) == 4:
        corners = approx
        found_corners = True
        break

if not found_corners:
    raise ValueError("未检测到四个角点，无法裁剪。")

# 对角点排序
corners = sorted(corners, key=lambda x: (x[0][1], x[0][0]))
top_points = sorted(corners[:2], key=lambda x: x[0][0])
bottom_points = sorted(corners[2:], key=lambda x: x[0][0])
sorted_corners = np.array([top_points[0], top_points[1], bottom_points[1], bottom_points[0]], dtype="float32")

# 计算裁剪后的图像尺寸
width = int(np.linalg.norm(sorted_corners[0][0] - sorted_corners[1][0]))
height = int(np.linalg.norm(sorted_corners[0][0] - sorted_corners[3][0]))
dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

# 透视变换
transform_matrix = cv2.getPerspectiveTransform(sorted_corners, dst_points)
cropped_image = cv2.warpPerspective(image, transform_matrix, (width, height))
cropped_image_borderless = cropped_image[5:-5, 5:-5]

# 第二步：颜色检测
color_ranges = {
    'F': ((92, 46, 87), (112, 66, 107)),
    '0': ((94, 51, 53), (114, 71, 73)),
    '1': ((93, 63, 80), (113, 83, 100)),
    '2': ((79, 65, 83), (99, 85, 103)),
    '3': ((120, 68, 89), (140, 88, 109)),
    '4': ((104, 63, 86), (124, 83, 106)),
    '5': ((60, 92, 86), (80, 112, 106)),
    '6': ((89, 73, 84), (109, 93, 104)),
}

num_rows, num_cols = 16, 30
matrix = [['H' for _ in range(num_cols)] for _ in range(num_rows)]
hsv_image = cv2.cvtColor(cropped_image_borderless, cv2.COLOR_BGR2HSV)
cell_width = hsv_image.shape[1] / num_cols
cell_height = hsv_image.shape[0] / num_rows

# 遍历网格并检测颜色
for row in range(num_rows):
    for col in range(num_cols):
        x_start = int(col * cell_width)
        y_start = int(row * cell_height)
        x_end = int((col + 1) * cell_width)
        y_end = int((row + 1) * cell_height)

        cell = hsv_image[y_start:y_end, x_start:x_end]
        avg_color = cv2.mean(cell)[:3]

        detected_value = 'H'
        for num, (lower, upper) in color_ranges.items():
            if all(lower[i] <= avg_color[i] <= upper[i] for i in range(3)):
                detected_value = num
                break

        matrix[row][col] = detected_value

# 构建并保存矩阵文件
output_str = f"{num_cols}x{num_rows}x99\n" + "\n".join("".join(row) for row in matrix)
with open("map.mine", "w") as file:
    file.write(output_str)

print("矩阵保存为文件: map.mine")
