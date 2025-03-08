import numpy as np
from PIL import Image
from pathlib import Path
import logging
from multiprocessing import Lock

logger = logging.getLogger(__name__)

def pixel_is_non_white(pixel):
    # Placeholder function to check if a pixel is non-white
    return pixel != (255, 255, 255)

def find_closest_color_index(pixel):
    # Placeholder function to find the closest color index
    return 0

def is_too_close(new_point, existing_points, min_distance):
    for point in existing_points:
        if np.linalg.norm(np.array(new_point) - np.array(point)) < min_distance:
            return True
    return False

def png2npy(png_path, output_dir, counter, lock, total_cnt):
    with lock:
        counter.value += 1
        current_count = counter.value
    npy_path = Path(output_dir) / Path(png_path).with_suffix(".npy").name
    logger.debug(f"{current_count}/{total_cnt} Processing {png_path} ...")
    logger.debug(f" -> {npy_path} ...")
    img = Image.open(str(png_path))
    minx, miny = (0, 0)
    width, height = img.size
    half_width = width / 2
    half_height = height / 2

    nodes, centers, classes, centers_norm, nns, instances = [], [], [], [], [], []
    kept_points = []  # To store the points that are kept after filtering

    # Define the minimum distance between two points to consider them "too close"
    min_distance = 10  # Adjust this value as needed

    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if not pixel_is_non_white(pixel):
                continue
            closest_index = find_closest_color_index(pixel)
            center = [x, y]
            center_norm = [(x - half_width) / half_width, (y - half_height) / half_height]

            # Check if this point is too close to any previously kept point with the same class
            if not is_too_close(center, [p for p, c in zip(kept_points, classes) if c == closest_index], min_distance):
                nodes.append([1, 1, 1, 1, 0, 0])
                centers.append(center)
                classes.append([closest_index])
                centers_norm.append(center_norm)
                nns.append([0])
                instances.append([-1])
                kept_points.append(center)

    # Reduce the number of data points to 1/10 of the original count
    reduction_factor = 10
    indices = np.arange(len(nodes))
    np.random.shuffle(indices)
    indices = indices[:len(nodes) // reduction_factor]

    nodes = [nodes[i] for i in indices]
    centers = [centers[i] for i in indices]
    classes = [classes[i] for i in indices]
    centers_norm = [centers_norm[i] for i in indices]
    nns = [nns[i] for i in indices]
    instances = [instances[i] for i in indices]

    data_gcn = {
        "nd_ft": nodes,
        "ct": centers,
        "cat": classes,
        "ct_norm": centers_norm,
        "nns": nns,
        "inst": instances,
    }
    np.save(npy_path, data_gcn)