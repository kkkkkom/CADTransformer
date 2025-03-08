from pathlib import Path
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def rename_file(file_path, tag0, tag):
    new_path = Path(str(file_path).replace(tag0, tag))
    logger.debug(f" new_path={new_path}")
    file_path.rename(new_path)

if __name__ == "__main__":
    splits = ["val", "test", "train"]
    for split in splits:
        all_file_paths = list(Path(f"./datasets/processed/png_rename/{split}").glob("*.png"))
        # all_file_paths = list(Path(f"./datasets/processed/svg_rename/{split}").glob("*.svg"))
        total_cnt = len(all_file_paths)
        for i, file_path in enumerate(all_file_paths):
            logger.debug(f"{i}/{total_cnt} Renaming {file_path} ->")
            # rename_file(file_path, ".svg", ".recolor.svg")
            # rename_file(file_path, ".recolor.svg", ".svg")
            rename_file(file_path, ".png", ".recolor.png")
            # rename_file(file_path, ".recolor.png", ".png")