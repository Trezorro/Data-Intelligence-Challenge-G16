"""Heatmap Plot.

Creates a heatmap plot of each agent on each grid.
"""
from Visualization import *
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pickle



def main():
    """Main function."""
    input_path, output_path = parse_args("Heatmap plot")

    algorithms = ("Monte Carlo", "Q Learning", "Sarsa")
    file_names = ("MC all.csv", "Q all.csv", "Sarsa all.csv")
    grid_names = ("stay_off_my_grass.grid", "experiment_house.grid")

    # Read grid files:
    grids = {"stay_off_my_grass.grid": None, "experiment_house.grid": None}

    for grid_name in grids.keys():
        with open(f"../Discrete-Simulations/grid_configs/{grid_name}",
                  "rb") as f:
            grid = pickle.load(f)
            if not hasattr(grid, "transposed_version"):
                grid.cells = grid.cells.T
            grids[grid_name] = grid

    histories = {"stay_off_my_grass.grid": {},
                 "experiment_house.grid": {}}



    for grid_name, grid in grids.items():
        for algorithm in algorithms:
            histories[grid_name][algorithm] = np.zeros_like(grid.cells)

    for algorithm, file_name in zip(algorithms, file_names):
        with open(input_path / file_name) as file:
            first_line = True
            for line in file:
                if first_line:
                    # Skip first line (i.e. header)
                    first_line = False
                    continue
                data = line.split(",", 6)
                grid_name = data[0][1:-1]  # Get rid of quotation marks
                history = data[6][1:-2]
                history = eval(history)
                for pos in history:
                    histories[grid_name][algorithm][pos] += 1

    # Normalize arrays and get rid of outside box
    for g in grid_names:
        for a in algorithms:
            histories[g][a] = histories[g][a] / histories[g][a].max()
            histories[g][a] = histories[g][a][-2:0:-1, 1:-1]

    # Now plot heatmaps
    # Start with subplot titles
    subplot_titles = []
    for grid_name in grid_names:
        for algorithm in algorithms:
            subplot_titles.append(f"{algorithm} on\n{grid_name}")

    fig = make_subplots(rows=2, cols=3,
                        subplot_titles=subplot_titles,
                        horizontal_spacing=0.1,
                        vertical_spacing=0)

    for row, grid_name in enumerate(grid_names):
        for col, algorithm in enumerate(algorithms):
            arr = histories[grid_name][algorithm]

            fig.add_trace(
                go.Heatmap(z=arr, xgap=1, ygap=1),
                row=row + 1, col=col + 1
            )
    fig.update_layout(
        height=700,
        width=1200,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)"
    )
    fig.update_xaxes(showticklabels=False,
                     dtick=1,
                     gridcolor="white")
    fig.update_yaxes(showticklabels=False,
                     scaleanchor="x",
                     dtick=1,
                     gridcolor="white")

    write_figure(fig, output_path / "heatmap.pdf")


if __name__ == '__main__':
    main()
