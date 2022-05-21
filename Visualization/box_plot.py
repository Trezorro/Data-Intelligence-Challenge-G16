"""Box Plot.

Creates box plots of the data from the experiments.
"""
from Visualization import *
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    """Main function."""
    input_path, output_path = parse_args("Box plot")
    df = pd.read_csv(input_path, sep=";")

    df = df.rename(columns={"cleaned": "Cleaned", "efficiency": "Efficiency"})

    grids = df["grid"].unique()
    metrics = ("Cleaned", "Efficiency", "F1 Score")
    algorithms = ("Monte Carlo", "Q Learning", "Sarsa")
    colors = ("red", "green", "blue")

    subplot_titles = [grids[0], grids[0], grids[0],
                      grids[1], grids[1], grids[1]]

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.1,
                        vertical_spacing=0.12)

    for row, grid in enumerate(grids):
        for col, metric in enumerate(metrics):
            for color, algorithm in zip(colors, algorithms):
                algo_df = df[df["Algorithm"] == algorithm]
                showlegend = col == 0 and row == 0

                fig.add_trace(
                    go.Box(y=algo_df[df["grid"] == grid][metric],
                           boxpoints="all",
                           jitter=0.3,
                           pointpos=-1.5,
                           name=algorithm,
                           legendgroup=algorithm,
                           marker={"color": color},
                           showlegend=showlegend),
                    row=row + 1, col=col + 1
                )

    # Change y axis titles
    for col, metric in enumerate(metrics):
        fig.update_yaxes(title_text=f"{metric} (%)", col=col + 1)

    fig.update_layout(
        legend={"x": 0.5,
                "y": -0.1,
                "xanchor": "center",
                "yanchor": "top",
                "orientation": "h"},
        height=800,
        width=1200
    )

    write_figure(fig, output_path / f"box_plots.pdf")


if __name__ == '__main__':
    main()
