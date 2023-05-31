from multiprocessing.sharedctypes import Value
from pathlib import Path
from collections.abc import Iterable

import dstnx


def move_text(new_file: Path, file: Path):
    new_file.write_text(file.read_text())


def move_bytes(new_file: Path, file: Path):
    new_file.write_bytes(file.read_bytes())


def move_files(files: Iterable[Path], outfp: Path, dtype: str = "text"):
    match dtype:
        case "text":
            move_fn = move_text
        case "bytes":
            move_fn = move_bytes
        case _:
            raise ValueError
    for file in files:
        new_file = outfp / file.name
        move_fn(new_file, file)
        print(f"Copied {file.name} to {new_file.parent}")


def move_tables():
    fp = dstnx.fp.REG_OUTPUT / "tex_tables"
    outfp = dstnx.fp.DATA / "fromdst"
    move_files(fp.glob("*"), outfp, dtype="text")


EXCLUDE_PLOTS = ["interact", "regerr", "roccurve"]


def is_valid(f: Path):
    return not (any(x in f.name for x in EXCLUDE_PLOTS) or f.is_dir())


def move_figures():
    fp = dstnx.fp.REG_PLOTS
    outfp = dstnx.fp.DATA / "fromdst"
    figures = (f for f in fp.glob("*") if is_valid(f))
    move_files(figures, outfp, dtype="bytes")


if __name__ == "__main__":
    move_tables()
    move_figures()
