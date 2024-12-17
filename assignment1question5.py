from PIL import Image
import numpy as np
from scipy.ndimage import convolve

def load_image(image_path):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(img)


def detect_edges(image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    edges_x = convolve(image, sobel_x)
    edges_y = convolve(image, sobel_y)
    edges = np.hypot(edges_x, edges_y)
    edges = np.uint8(edges)
    return edges


def detect_flag_region(edges):
    rows, cols = edges.shape
    threshold = 100
    flag_region = edges > threshold
    rows_with_flag = np.any(flag_region, axis=1)
    cols_with_flag = np.any(flag_region, axis=0)

    top, bottom = np.where(rows_with_flag)[0][[0, -1]]
    left, right = np.where(cols_with_flag)[0][[0, -1]]

    return top, bottom, left, right


def check_flag_colors(image, top, bottom, left, right):
    flag_area = image[top:bottom, left:right]
    mid_row = (top + bottom) // 2
    top_half = flag_area[:mid_row - top, :]
    bottom_half = flag_area[mid_row - top:, :]
    top_color = np.mean(top_half)
    bottom_color = np.mean(bottom_half)
    if top_color > bottom_color:
        return "Poland Flag"
    else:
        return "Indonesia Flag"


def main(image_path):
    img = Image.open(image_path)
    grayscale_image = load_image(image_path)
    edges = detect_edges(grayscale_image)
    top, bottom, left, right = detect_flag_region(edges)
    result = check_flag_colors(np.array(img), top, bottom, left, right)
    return result
image_path = 'flag.PNG'
flag = main(image_path)
print(f"The given image is the {flag}.")
