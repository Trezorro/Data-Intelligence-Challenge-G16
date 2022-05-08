"""Box Plots.

Creates box plots of the data from the experiments.
"""
import plotly.express as px
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Box Plots")
    parser.add_argument("input", type=Path, help="Input file")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    df = pd.read_excel(args.input, sheet_name="Sheet1")

    for metric in ("Clean", "Efficiency", "F-Score"):
        fig = px.box(df, x="Agent", y=metric, color="Agent", facet_col="Label",
                     facet_col_wrap=4, points="all")
        fig.update_layout(showlegend=False)
        fig.show()


if __name__ == '__main__':
    main()
