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


def layout_fig_legend_titles(fig):
    # Change subplot title position
    for annotation in fig["layout"]["annotations"]:
        annotation["y"] = 1.02

    # Change y axis titles
    fig.update_yaxes(title_text="Portion of room cleaned (%)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="F<sub>1</sub> Score", row=3, col=1)

    fig.update_layout(
        legend={"x": 0.5,
                "y": -0.1,
                "xanchor": "center",
                "yanchor": "top",
                "orientation": "h"},
        height=900,
        width=700
    )
    return fig


def write_figure(figure: Figure, output_path: Path):
    figure.write_image(output_path)

    # Added this to get rid of "Loading" box at the bottom
    sleep(1)
    figure.write_image(output_path)
