import logging
from pathlib import Path
from multiprocessing import Pool, Manager, Lock
from functools import partial

from _utils_dataset import *

split="train"

def svg2png_colored(svg_path, counter, lock, total_cnt):
    with lock:
        counter.value += 1
        current_count = counter.value
    out_path = Path(f"./processed/png_colored/{split}") / svg_path.with_suffix(".png").name
    print(f"{current_count}/{total_cnt}, Processing svg_path={svg_path} ...")
    print(f" out_path={out_path} ...")
    svg2png(svg_path, out_path, scale=7, sleep=0)
    # time.sleep(0.1)

def init_worker(counter_, lock_):
    global counter, lock
    counter = counter_
    lock = lock_

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()

    all_svg_paths = list(Path(f"./processed/svg_recolor/{split}").glob("*.svg"))[:]
    # all_svg_paths = list(
    #     Path(f"./a/flooplanced_v1/svg_raw/{split}/{split}/svg_gt").glob("*.svg")
    # )
    total_cnt = len(all_svg_paths)
    out_dir = Path(f"./processed/png_colored/{split}")
    out_dir.mkdir(exist_ok=True, parents=True)

    manager = Manager()
    counter = manager.Value("i", 0)  # Shared counter
    lock = manager.Lock()  # Shared lock for synchronization

    # Initialize the worker processes with the shared counter and lock
    with Pool(32, initializer=init_worker, initargs=(counter, lock)) as p:
        try:
            # Use partial to bind counter, lock, and total_cnt
            func = partial(
                svg2png_colored, counter=counter, lock=lock, total_cnt=total_cnt
            )
            p.map(func, all_svg_paths)
        except KeyboardInterrupt:
            print("....\nCaught KeyboardInterrupt, terminating workers")
            p.terminate()
        else:
            p.close()
        p.join()