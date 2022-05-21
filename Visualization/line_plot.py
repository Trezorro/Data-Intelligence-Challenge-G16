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

    subplot_titles = []

    # Make subplot titles:
    for var in variables:
        for metric in metrics:
            subplot_titles.append(f"{metric} vs. {var}")

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.1,
                        vertical_spacing=0.12)

    for row, (var, df) in enumerate(zip(variables, [eps_df, gamma_df])):
        for col, metric in enumerate(metrics):
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
    for row, var in enumerate(variables):
        fig.update_xaxes(title_text=var, row=row + 1)


    # Change y axis title
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
    write_figure(fig, output_path / "line_plots.pdf")


if __name__ == '__main__':
    main()
