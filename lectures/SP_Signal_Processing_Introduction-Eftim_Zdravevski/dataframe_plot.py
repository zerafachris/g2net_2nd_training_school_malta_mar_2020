import plotly.graph_objs as go
from plotly.offline import plot  # download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.figure_factory as ff
# import plotly.plotly as py
# from plotly import tools
import os
# import datetime
import pandas as pd
from typing import List


def generate_scatter_plot(dataframe: pd.DataFrame, column_name_list: List[str], out_file: str = 'chart.html',
                          title: str = None, mode='lines+markers'):
    """

    mode: Any combination of ['lines', 'markers', 'text'] joined with '+' characters
    """
    data = list()
    for column_name in column_name_list:
        data.append(
            go.Scatter(
                x=dataframe.index,
                y=dataframe[column_name],
                name=column_name,
                mode=mode
            )
        )
    if not title:
        title = os.path.basename(out_file).replace('.html', '')
    if title is not None:
        layout = go.Layout(title=title)
        fig = go.Figure(data=data, layout=layout)
    else:
        fig = go.Figure(data=data)
    if out_file is None:
        fig.show()
    else:
        plot(fig, filename=out_file)


def generate_table(dataframe: pd.DataFrame, column_name_list: List[str] = None, out_file: str = 'table.html'):
    if column_name_list is None:
        table = ff.create_table(dataframe, index=True)
    else:
        table = ff.create_table(dataframe[column_name_list], index=True)

    if out_file is None:
        table.show()
    else:
        plot(table, filename=out_file)
