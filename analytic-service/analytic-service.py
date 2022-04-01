import dash
import dash_core_components as dcc
import dash_html_components as html
import glob
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd


def generate_table(dataframe, max_rows=10):
    return html.Table(
        [
            html.Thead(html.Tr([html.Th(col) for col in dataframe.columns])),
            html.Tbody(
                [
                    html.Tr(
                        [
                            html.Td(dataframe.iloc[i][col])
                            for col in dataframe.columns
                        ]
                    )
                    for i in range(min(len(dataframe), max_rows))
                ]
            ),
        ]
    )


app = dash.Dash(__name__)


# Build a dataframe with test data
parquet_files = glob.glob(os.path.join("/data/raw", "*.parquet.gzip"))
df_list = []
for file in parquet_files:
    df_ = pd.read_parquet(file)
    df_list.append(df_)
df = pd.concat(df_list)
df.drop_duplicates(inplace=True)
df["time"] = pd.to_datetime(df.time, infer_datetime_format=False)
df.set_index("time", inplace=True)
df.sort_index(inplace=True)

# Build a figure
fig = go.Figure(
    data=[
        go.Scatter(
            x=df.index, y=df.close
        )
    ]
)
app.layout = html.Div(
    [
        html.H1("Signal Pricing Analytic Service"),
        html.Div(os.getenv("SYMBOL")),
        dcc.Graph(id="signal-pricing-analytic-service", figure=fig),
    ],
)


if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=False, port=8000, use_reloader=True)
