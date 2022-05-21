from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
from time import sleep
from plotly.graph_objects import Figure



def parse_args(description: str) -> Tuple[Path, Path]:
    """Parse command line arguments."""
    parser = ArgumentParser(description=description)
    parser.add_argument("input", type=Path, help="Input file")
    parser.add_argument("output", type=Path, help="Plot output directory")

    args = parser.parse_args()
    if not args.output.exists():
        args.output.mkdir()

    return args.input, args.output


def write_figure(figure: Figure, output_path: Path):
    figure.write_image(output_path)

    # Added this to get rid of "Loading" box at the bottom
    sleep(1)
    figure.write_image(output_path)
