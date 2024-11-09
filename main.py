import cv2
import numpy as np

def process_image(image_path):
    # 读取图像并进行预处理
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path '{image_path}' not found or cannot be loaded.")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 20, 100)
    edges = cv2.erode(cv2.dilate(edges, np.ones((3, 3), np.uint8)), None)

    # 寻找四边形轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    corners = next((cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True) for c in contours if len(cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)) == 4), None)

    if corners is None:
        print("未检测到四个角点，无法裁剪。")
        return None

    # 排序并裁剪图像
    corners = sorted(corners, key=lambda x: (x[0][1], x[0][0]))
    top, bottom = sorted(corners[:2], key=lambda x: x[0][0]), sorted(corners[2:], key=lambda x: x[0][0])
    sorted_corners = np.array([top[0], top[1], bottom[1], bottom[0]], dtype="float32")

    width, height = np.linalg.norm(sorted_corners[0] - sorted_corners[1]), np.linalg.norm(sorted_corners[0] - sorted_corners[3])
    dst_points = np.array([[0.0, 0.0], [width-1.0, 0.0], [width-1.0, height-1.0], [0.0, height-1.0]], dtype="float32")

    cropped_image = cv2.warpPerspective(image, cv2.getPerspectiveTransform(sorted_corners, dst_points), (int(width), int(height)))
    cv2.imwrite("temmap.png", cropped_image)
    return cropped_image[5:-5, 5:-5]  # 返回去除边框后的图像

def detect_colors(image_path):
    # 加载图像并转换为 HSV
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 颜色范围定义
    color_ranges = {
        'F': ((92.0, 46.0, 87.0), (112.0, 66.0, 107.0)),
        '0': ((94.0, 51.0, 53.0), (114.0, 71.0, 73.0)), 
        '1': ((93.0, 63.0, 80.0), (113.0, 83.0, 100.0)),  
        '2': ((79.0, 65.0, 83.0), (99.0, 85.0, 103.0)),   
        '3': ((120.0, 68.0, 89.0), (140.0, 88.0, 109.0)),
        '4': ((104.0, 63.0, 86.0), (124.0, 83.0, 106.0)),
        '5': ((60.0, 92.0, 86.0), (80.0, 112.0, 106.0)),
        '6': ((89.0, 73.0, 84.0), (109.0, 93.0, 104.0)),
    }

    # 初始化矩阵
    rows, cols = 16, 30
    matrix = [[('H', (0.0, 0.0, 0.0)) for _ in range(cols)] for _ in range(rows)]

    # 图像尺寸和网格分割
    height, width = image.shape[:2]
    cell_width, cell_height = width / cols, height / rows  # 使用浮点数

    # 排除颜色范围
    excluded_colors = np.array([[65.0, 58.0, 50.0], [104.0, 68.0, 56.0], [105.0, 87.0, 41.0]])
    lower_bounds, upper_bounds = excluded_colors - 10.0, excluded_colors + 10.0  # 使用浮点数

    # 遍历每个网格单元并进行颜色检测
    for row in range(rows):
        for col in range(cols):
            x_start, y_start = col * cell_width, row * cell_height
            x_end, y_end = (col + 1) * cell_width, (row + 1) * cell_height
            cell = hsv_image[int(y_start):int(y_end), int(x_start):int(x_end)]  # 截取单元格区域

            # 计算平均颜色值
            avg_color = cv2.mean(cell)[:3]

            # 根据颜色匹配
            detected_value = 'H'
            for num, (lower, upper) in color_ranges.items():
                if all(lower[i] <= avg_color[i] <= upper[i] for i in range(3)):
                    detected_value = num
                    break

            matrix[row][col] = (detected_value, avg_color)

    # 输出矩阵并保存到文件
    output_str = f"{cols}x{rows}x99\n" + "\n".join("".join(cell[0] for cell in row) for row in matrix) + "\n"
    with open("map.mine", "w") as file:
        file.write(output_str)

    print(f"矩阵保存为文件: map.mine")

# 执行流程
cropped_image = process_image("map.png")
if cropped_image is not None:
    detect_colors("crop_map.png")
