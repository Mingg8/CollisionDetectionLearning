import plotly as py
from plotly import graph_objs as go
import numpy as np

colors = ['darkblue', 'teal']

class Plot(object):
    def __init__(self, filename):
        """
        Create a plot
        :param filename: filename
        """
        self.filename = "visualizations/" +filename + ".html"
        self.data = []
        self.layout = {'title': 'Plot',
                       'showlegend': False
                       }

        self.fig = {'data': self.data,
                    'layout': self.layout}


    def plot_3d_pnts(self, pnts, colors = None):
        """
        Plot 3D trees
        :param trees: trees to plot
        """
        pnts = np.array(pnts)
        # for i in range(np.shape(pnts)[0]):
        trace = go.Scatter3d(
            x=pnts[:,0],
            y=pnts[:,1],
            z=pnts[:,2],
            mode='markers',
            marker=dict(
                size=12,
                color=colors,                # set color to an array/list of desired values
                colorscale='Viridis',   # choose a colorscale
                opacity=0.8,
                colorbar=dict(thickness=20)
            )
        )
        self.data.append(trace)


    def plot_2d_pnts(self, pnts, colors = None):
        """
        Plot 3D trees
        :param trees: trees to plot
        """
        trace = go.Scatter(
            x=pnts[:,0],
            y=pnts[:,1],
            # mode='line',
            # marker=dict(
            #     size=12,
            #     color=colors,                # set color to an array/list of desired values
            #     colorscale='Viridis',   # choose a colorscale
            #     opacity=0.8
            # )
        )
        self.data.append(trace)

    def draw(self, auto_open=True):
        """
        Render the plot to a file
        """
        py.offline.plot(self.fig, filename=self.filename, auto_open=auto_open)