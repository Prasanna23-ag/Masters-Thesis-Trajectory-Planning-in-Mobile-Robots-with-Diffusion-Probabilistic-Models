import cv2
import yaml
import numpy as np

def load_maze_from_yaml_and_pgm(yaml_path, pgm_path):
    # Load metadata
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    resolution = data['resolution']  # meters per pixel
    origin = data['origin']          # [x, y, theta]

    # Load map image
    img = cv2.imread(pgm_path, cv2.IMREAD_GRAYSCALE)
    maze_arr = (img < 250).astype(np.int32)  # 1 = obstacle, 0 = free

    # Compute bounds in world coordinates
    height, width = maze_arr.shape
    x_min = origin[0]
    x_max = origin[0] + width * resolution
    y_min = origin[1]
    y_max = origin[1] + height * resolution
    bounds = (x_min, x_max, y_min, y_max)

    return maze_arr, bounds
