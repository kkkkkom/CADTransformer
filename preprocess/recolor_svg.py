import re
import logging
from functools import partial
from pathlib import Path
from multiprocessing import Pool, Manager

import anno_config
from _utils_dataset import *

color_pallete = anno_config.color_pallete

def recolor_svg(svg_path, out_dir, counter, lock, total_cnt):
    with lock:
        counter.value += 1
        current_count = counter.value
        out_path = out_dir / svg_path.with_suffix(".recolor.svg").name
        print(f"{current_count}/{total_cnt}, Processing svg_path={svg_path} ...")
        print(f" out_path={out_path} ...")
        with open(svg_path) as f:
            with open(out_path, "w") as fw:
                for line in f:
                    # logger.debug(f"line={line}")
                    semantic_match = re.search(r'semanticId="(\d+)"', line)
                    if not semantic_match:
                        fw.write(line)
                        continue
                    semantic_id = int(semantic_match.group(1))
                    new_line = re.sub(
                        r"rgb\((.*?)\)", f"rgb{color_pallete[semantic_id]}", line)
                    fw.write(new_line)

# splits = ["val"]
splits = ["val", "test", "train"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    for split in splits:
        all_svg_paths = list(
            Path(f"./data/floorplancad_v1/svg_raw/{split}/{split}/svg_gt").glob("*.svg")
        )
        out_dir = Path(f"./processed/svg_recolor/{split}")
        out_dir.mkdir(exist_ok=True, parents=True)

        total_cnt = len(all_svg_paths)
        manager = Manager()
        counter = manager.Value("i", 0)  # Shared counter
        lock = manager.Lock()  # Shared lock for synchronization

        # Initialize the worker processes with the shared counter and lock
        with Pool(32) as p:
            try:
                # Use partial to bind counter, lock, and total_cnt
                func = partial(
                    recolor_svg,
                    out_dir=out_dir,
                    counter=counter,
                    lock=lock,
                    total_cnt=total_cnt,
                )
                p.map(func, all_svg_paths)
            except KeyboardInterrupt:
                print("....\nCaught KeyboardInterrupt, terminating workers")
                p.terminate()
            else:
                p.close()
            p.join()