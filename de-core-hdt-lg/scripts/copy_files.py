import glob
import sys
from pathlib import Path

import typer


def main(stem: str, ext: str, input_dir: Path, output_dir: Path):
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, 'w') as fout:
        for filename in glob.glob(str(input_dir.resolve()) + f"/*-{stem}*.{ext}"):
            for line in open(filename):
                fout.write(line)


if __name__ == "__main__":
    typer.run(main)
