"""Web app.

Provides a way to visualize what the robot is doing in a web browser.
"""
from dash import Dash, html, dcc
import plotly.graph_objects as go
import pandas as pd
import agent_configs
from pathlib import Path
from typing import Optional


def make_grid(grid_file: Optional[Path]) -> go.Figure:
    """Reads a Grid file and returns it as a Figure.

    Args:
        grid_file: Grid file to be drawn. If none is given, returns an empty
            grid.
    """
    if grid_file is None:
        fig = go.Figure(layout={"width": 600,
                                "height": 600})
    return fig



def main():
    app = Dash("rl-vacuum-cleaner")

    maps = [i.stem for i in Path("map_configs").glob("*.grid")]

    # Temporary data
    # df = pd.DataFrame({
    #     "Fruit": ["Apples", "Oranges", "Grapes", "Apples", "Oranges",
    #               "Grapes"],
    #     "Amount": [4, 1, 2, 2, 4, 5],
    #     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
    # })

    # fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

    # Layout the app
    colors = {"surface": "#F9F9F9",
              "text": "#000000"}
    fonts = {"san_serif": ["Open Sans", "HelveticaNeue", "Helvetica Neue",
                           "Helvetica", "Arial", "sans-serif"]}
    # fig.update_layout(
    #     plot_bgcolor=colors["surface"],
    #     paper_bgcolor=colors["surface"],
    #     font_color=colors["text"],
    #     width=600,
    #     height=600
    # )

    fig = make_grid(None)

    button_style = {
        "padding": 10,
        "margin": 10,
        "fontSize": 18,
        "borderRadius": "5px",
        "backgroundColor": "white",
        "border": "1px solid #CCCCCC"
    }
    label_style = {
        "padding": 10,
    }
    app.layout = html.Div(
        style={
            'backgroundColor': colors["surface"],
            "fontFamily": fonts["san_serif"],
            "padding": 10,
        },
        children=[
            html.H1(
                children='The Reinforcement Learning Vacuum Cleaner',
                style={
                    "textAlign": "center",
                    "color": colors["text"]
                }
            ),  # Title
            html.Div(
                children="The Vacuum Cleaner... of the Future!",
                style={
                    "textAlign": "center",
                    "color": colors["text"]
                }
            ),  # Subtitle
            html.Div(
                style={
                    "padding": 10,
                    "display": "flex",
                    "flexDirection": "row"
                },
                children=[
                    html.Div(
                        style={
                            "padding": 10,
                            "minWidth": 150,
                            "flex": 1,
                            "display": "flex",
                            "flexDirection": "column"
                        },
                        children=[
                            html.Label(
                                "Map Selection",
                                style=label_style
                            ),  # Map Selection Label
                            dcc.Dropdown(
                                maps,
                                searchable=False
                            ),  # Map selection dropdown
                            html.Label(
                                "Robot Selection",
                                style=label_style
                            ),  # Robot Selection label
                            dcc.Dropdown(
                                agent_configs.__all__,
                                searchable=False
                            ),  # Robot selection dropdown
                            dcc.Input(
                                id="num-robots",
                                type="number",
                                placeholder="Number of robots to spawn",
                                min=1,
                                step=1,
                                style=button_style
                            ),
                            html.Button(
                                "Spawn Robot",
                                id="spawn-robot",
                                style=button_style
                            ),  # Spawn Robot button
                            html.Button(
                                "Start Simulation",
                                id="start-simulation",
                                style=button_style
                            ),  # Start simulation button
                            html.Button(
                                "Stop Simulation",
                                id="stop-simulation",
                                style=button_style
                            ),  # Stop simulation button
                            html.Label(
                                "Simulation Speed",
                                style=label_style
                            ),
                            dcc.Slider(
                                id="simulation-agent_speed",
                                min=1,
                                max=5,
                                step=1,
                                value=1,
                                marks={
                                    1: "1x",
                                    2: "2x",
                                    3: "3x",
                                    4: "4x",
                                    5: "5x"
                                },
                            ),  # Slider
                        ],
                    ),  # Settings div
                    html.Div(
                        style={
                            "padding": 10,
                            "flex": 1
                        }, children=[
                            dcc.Graph(
                                id='example-graph',
                                figure=fig,
                                style={
                                    "minimumHeight": 800
                                }  # Graph Style
                            )  # Graph
                        ]
                    )  # Graph div
                ],  # End children
            ),  # Contents div
        ]
    )
    app.run_server(debug=True)


if __name__ == '__main__':
    main()
