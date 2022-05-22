"""Line plot.

Plots effect of epsilon and gamma on cleaned, efficiency, and F1 score.
"""
from Visualization import *
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def main():
    input_path, output_path = parse_args("Line plot")

    eps_df = pd.read_csv(input_path / "epsilon_data.csv", sep=";")
    gamma_df = pd.read_csv(input_path / "gamma_data.csv", sep=";")

    variables = ("Epsilon", "Gamma")
    metrics = ("Cleaned", "Efficiency", "F1 Score")
    algorithms = ("Monte Carlo", "Q Learning", "Sarsa")
    colors = ("red", "green", "blue")

    subplot_titles = ["Effect of epsilon on metric", "Effect of gamma on metric"]

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.12,
                        vertical_spacing=0.05)

    for col, (var, df) in enumerate(zip(variables, [eps_df, gamma_df])):
        for row, metric in enumerate(metrics):
            for color, algorithm in zip(colors, algorithms):
                algo_df = df[df["Algorithm"] == algorithm]
                showlegend = col == 0 and row == 0

                fig.add_trace(
                    go.Scatter(x=algo_df[var], y=algo_df[metric],
                               name=algorithm, legendgroup=algorithm,
                               marker={"color": color}, showlegend=showlegend),
                    row=row + 1, col=col + 1
                )
    # Change y axis range
    fig.update_yaxes(range=[-5, 100])

    # Change x axis title
    fig.update_xaxes(title_text=variables[0], row=3, col=1)
    fig.update_xaxes(title_text=variables[1], row=3, col=2)

    fig = layout_fig_legend_titles(fig)

    write_figure(fig, output_path / "line_plots.pdf")


if __name__ == '__main__':
    main()
