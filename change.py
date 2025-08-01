
import cv2

# 读取图片
image = cv2.imread('map-2025.png')

# 设置新的分辨率 (宽, 高)
new_width = 2800
new_height = 1500

# 调整图片大小
resized_image = cv2.resize(image, (new_width, new_height))

# 保存调整后的图片
cv2.imwrite('mapRMUC.jpg', resized_image)

# 显示图片（可选）
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
