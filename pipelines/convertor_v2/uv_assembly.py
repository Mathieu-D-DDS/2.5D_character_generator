import cv2
import numpy as np

def assemble_uv(front_path, back_path, left_path, right_path, output_uv_path):
    front = cv2.imread(front_path)
    back = cv2.imread(back_path)
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    h, w, _ = front.shape
    uv_map = np.zeros((h*2, w*2, 3), dtype='uint8')
    uv_map[0:h,   0:w]   = front
    uv_map[0:h,   w:2*w] = back
    uv_map[h:2*h, 0:w]   = left
    uv_map[h:2*h, w:2*w] = right
    cv2.imwrite(output_uv_path, uv_map)