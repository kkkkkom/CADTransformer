import logging
import multiprocessing
from datetime import datetime

from PIL import Image, ImageDraw
import math
import numpy as np
from pathlib import Path
from functools import partial, lru_cache
from multiprocessing import Pool, Manager

from _utils_dataset import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.CRITICAL)

color_palette = anno_config.color_pallete

@lru_cache(maxsize=2048)
def euclidean_distance(color1, color2):
    """Calculates the Euclidean distance between two RGB colors."""
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))

@lru_cache(maxsize=2048)
def find_closest_color_index(pixel_color):
    """Finds the color index of the closest color in the palette."""
    min_distance = float("inf")
    closest_index = None

    for index, palette_color in color_palette.items():
        distance = euclidean_distance(pixel_color, palette_color)
        if distance < min_distance:
            min_distance = distance
            closest_index = index

    return closest_index

@lru_cache(maxsize=2048)
def pixel_is_non_white(pixel):
    if isinstance(pixel, tuple):
        if len(pixel) == 3:  # RGB
            if pixel != (255, 255, 255):
                return True
        elif len(pixel) == 4:  # RGBA
            if pixel[:3] != (255, 255, 255):  # ignore alpha
                return True
    elif isinstance(pixel, int):  # L (grayscale)
        if pixel != 255:
            return True
    return False

def is_too_close(new_point, existing_points, min_distance):
    for point in existing_points:
        if np.linalg.norm(np.array(new_point) - np.array(point)) < min_distance:
            return True
    return False

def is_point_too_close(new_point, existing_point, min_distance):
    if np.linalg.norm(np.array(new_point) - np.array(existing_point)) < min_distance:
        return True
    return False

def visualize_npy(npy_data, png_path=None, show=False):
    image = Image.new("RGB", (980, 980), "white")
    draw = ImageDraw.Draw(image)

    # Draw dots
    dot_radius = 3
    for x, y in npy_data["center"]:
        draw.ellipse((x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius), fill="black")

    # Save the image
    if png_path is not None:
        image.save(png_path)

    # Show the image (optional)
    if show:
        image.show()


def png2npy(png_path, output_dir, counter, lock, total_cnt):
    with lock:
        counter.value += 1
        current_count = counter.value
    npy_path = Path(output_dir) / Path(png_path).with_suffix(".npy").name
    t0 = datetime.now()
    logger.debug(f"{current_count}/{total_cnt} Processing {png_path} ...")
    logger.debug(f"{current_count}/{total_cnt}      -> {npy_path} ...")
    img = Image.open(str(png_path))
    minx, miny = (0, 0)
    width, height = img.size
    half_width = width / 2
    half_height = height / 2

    nodes, centers, classes, centers_norm, nns, instances = [], [], [], [], [], []
    for x in range(width):
        for y in range(height):
            pixel = img.getpixel((x, y))
            if not pixel_is_non_white(pixel):
                continue
            closest_index = find_closest_color_index(pixel)
            nodes.append([1, 1, 1, 1, 0, 0])
            centers.append([x, y])
            classes.append([closest_index])
            centers_norm.append(
                [(x - half_width) / half_width, (y - half_height) / half_height]
            )
            nns.append([0])
            instances.append([-1])

    ### filter to reduce density ###
    len0 = len(nodes)
    min_distance = 8
    max_cnt = len0 / 8
    kept_indices = list(range(len0))
    for _ in range(16):
        if min_distance > width/2:
            break
        if len(kept_indices) < max_cnt:
            break
        logger.debug(f"{current_count}/{total_cnt}      min_distance={min_distance}")
        new_kept_indices = []
        for i in kept_indices:
            keep = True
            for j in new_kept_indices:
                if classes[i]!=classes[j]:
                    continue
                if is_point_too_close(centers[i], centers[j], min_distance):
                    keep = False
                    break
            if keep:
                new_kept_indices.append(i)
        # nodes = [nodes[i] for i in kept_indices]
        # centers = [centers[i] for i in kept_indices]
        # classes = [classes[i] for i in kept_indices]
        # centers_norm = [centers_norm[i] for i in kept_indices]
        # nns = [nns[i] for i in kept_indices]
        # instances = [instances[i] for i in kept_indices]
        kept_indices = new_kept_indices
        min_distance*=2
    t1 = datetime.now()
    logger.debug(f"{current_count}/{total_cnt} Done filtering {t1-t0}.")
    nodes = [nodes[i] for i in kept_indices]
    centers = [centers[i] for i in kept_indices]
    classes = [classes[i] for i in kept_indices]
    centers_norm = [centers_norm[i] for i in kept_indices]
    nns = [nns[i] for i in kept_indices]
    instances = [instances[i] for i in kept_indices]
    #########

    data_gcn = {
        "nd_ft": nodes,
        "ct": centers,
        "cat": classes,
        "ct_norm": centers_norm,
        "nns": nns,
        "inst": instances,
    }
    np.save(npy_path, data_gcn)

if __name__ == "__main__":
    splits = ["train"]

    start, end = 0, None
    for split in splits:
        all_png_paths = sorted(list(Path(f"./processed/png_colored/{split}").glob("*.png")))[start:end]

        total_cnt = len(all_png_paths)
        out_dir = Path(f"./processed/npy_pixeled/{split}")
        out_dir.mkdir(exist_ok=True, parents=True)

        manager = Manager()
        counter = manager.Value("i", start)  # Shared counter
        lock = manager.Lock()  # Shared lock for synchronization

        # Initialize the worker processes with the shared counter and lock
        # with Pool(32, init_worker) as p:
        with Pool(multiprocessing.cpu_count(), init_worker) as p:
            try:
                # Use partial to bind counter, lock, and total_cnt
                func = partial(
                    png2npy,
                    output_dir=out_dir,
                    counter=counter,
                    lock=lock,
                    total_cnt=total_cnt,
                )
                p.map(func, all_png_paths)
            except KeyboardInterrupt:
                print("....\nCaught KeyboardInterrupt, terminating workers")
                p.terminate()
            else:
                p.close()
                p.join()