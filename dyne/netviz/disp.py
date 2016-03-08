"""
Adjacency display classes for visualization
"""
# Author: Ankit Khambhati
# License: BSD 3-Clause

from __future__ import division
import os
import csv
from ...common.pipe import NetvizPipe
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.ion()


def time_to_datetime(t):
    hms = time.strftime('%H:%M:%S', time.gmtime(t))
    ms = ('%0.3f' % (t-math.floor(t)))[1:]

    return hms + ms


class AdjacencyDisplay(NetvizPipe):
    """
    AdjacencyDisplay class for plotting the dynamic adjacency matrix

    Parameters
    ----------
    adj_lim: tuple(float), shape: (2,)
        Limits for colormap
    """

    def __init__(self, pipe_name=None, adj_lim=None,
                 cache=False):
        super(AdjacencyDisplay, self).__init__(pipe_name, cache)

        assert type(adj_lim) == tuple or type(adj_lim) == list
        assert len(adj_lim) == 2
        self.adj_lim = adj_lim

        # Setup figure
        plt.ion()
        self.font_size_ = 14
        self.fig_ = plt.figure()
        self.fig_.set_facecolor('w')
        self.ax_ = self.fig_.add_subplot(111)
        self.im_ = self.ax_.matshow([[1, 1], [1, 1]],
                                    vmin=adj_lim[0], vmax=adj_lim[1])
        self.ax_.set_axis_off()

    def _func_def(self, adj, sec):
        """
        Update the current figure with the adjacency matrix
        """
        elapsed_time = time_to_datetime(sec)

        self.im_.set_data(adj)
        self.ax_.set_title('Time: %s' % elapsed_time)
        plt.draw()


class GraphDisplay(NetvizPipe):
    """
    GraphDisplay class for plotting the dynamic graph connectivity

    Parameters
    ----------
    adj_lim: list(2-tuple), shape: (n_connect_class,)
        List of limits for colormap (must be between 1 and 4)

    sensor_map_path: str
        Path to PNG file containing the image of the sensor map

    sensor_coords_path: str
        Path to MAT file containing the sensor coordinates with respect to
        sensor_map (in variable called 'coords')

    Attributes
    ----------
    sensor_coords_: list(tuple), shape: (n_node,), tuple-shape: (2,)
        Two-dimensional coordinates of the sensors

    ax_sensor_map_: AxesImage
        Object containing the sensor_map image

    ax_sensor_node_: list(PathCollection), shape: (n_node,)
        List of objects containing node representations
    """

    def __init__(self, pipe_name=None, adj_lim=None, sensor_map_path=None,
                 sensor_coords_path=None, cache=False):
        super(GraphDisplay, self).__init__(pipe_name, cache)

        assert type(adj_lim) == list
        assert len(adj_lim) >= 1 and len(adj_lim) <= 4
        self.adj_lim = adj_lim
        self.n_ax_ = len(adj_lim)

        if not os.path.exists(sensor_map_path):
            raise IOError('%s does not exist' % sensor_map_path)
        if not os.path.exists(sensor_coords_path):
            raise IOError('%s does not exist' % sensor_coords_path)

        # Load sensor_map
        sensor_map_img = mpimg.imread(sensor_map_path)

        # Load Coords
        sensor_coords = []
        sensor_coords_file = csv.reader(open(sensor_coords_path, 'rb'),
                                        delimiter=',')
        for row in sensor_coords_file:
            sensor_coords.append(row)
        self.sensor_coords_ = sensor_coords
        self.triu_idx_ = np.triu_indices(len(self.sensor_coords_), k=1)
        n_connects = len(self.triu_idx_[0])

        # Setup figure
        self.font_size_ = 14
        self.fig_ = plt.figure(figsize=(16, 10))
        self.fig_.set_facecolor('w')
        self.fig_.canvas.set_window_title(self.pipe_name)

        # Iterate over each subplot axis (one for each threshold group)
        # Initialize list for each figure-level object
        self.ax_ = []
        self.ax_sensor_map_ = []
        self.ax_sensor_node_ = []
        self.ax_sensor_connect_ = []
        for ax_idx in xrange(self.n_ax_):
            if self.n_ax_ < 3:
                self.ax_.append(self.fig_.add_subplot(1, self.n_ax_, ax_idx+1))
            else:
                self.ax_.append(self.fig_.add_subplot(2, 2, ax_idx+1))
            self.ax_[ax_idx].set_axis_off()

            # Plot sensor map image
            self.ax_sensor_map_.append(self.ax_[ax_idx].imshow(sensor_map_img))

            # Plot sensor node coordinates
            ax_sensor_node = []
            for coords in self.sensor_coords_:
                ax_sensor_node.append(
                    self.ax_[ax_idx].scatter(coords[0], coords[1],
                                             s=75, color=[0, 0, 0]))
            self.ax_sensor_node_.append(ax_sensor_node)

            # Plot sensor connections
            ax_sensor_connect = []
            for connect_idx in xrange(n_connects):
                x1 = self.sensor_coords_[self.triu_idx_[0][connect_idx]][0]
                y1 = self.sensor_coords_[self.triu_idx_[0][connect_idx]][1]
                x2 = self.sensor_coords_[self.triu_idx_[1][connect_idx]][0]
                y2 = self.sensor_coords_[self.triu_idx_[1][connect_idx]][1]

                ax_sensor_connect.append(self.ax_[ax_idx].plot((x1, x2),
                                                               (y1, y2),
                                                               alpha=0.0,
                                                               linewidth=2.0))
            self.ax_sensor_connect_.append(ax_sensor_connect)

            # Set title for axis
            self.ax_[ax_idx].set_title('Threshold: %0.3f -- %0.3f' %
                                       (self.adj_lim[ax_idx][0],
                                        self.adj_lim[ax_idx][1]),
                                       fontsize=self.font_size_-2)

        self.fig_.subplots_adjust(left=0.001,
                                  right=0.999,
                                  bottom=0.001,
                                  top=0.899,
                                  wspace=0.001)

        self.fig_.show()

        # Get background state
        self.fig_.canvas.draw()
        self.ax_background_ = [
            self.fig_.canvas.copy_from_bbox(
                ax.bbox) for ax in self.ax_]
        self.fig_background_ = self.fig_.canvas.copy_from_bbox(self.fig_.bbox)

        elapsed_time = time_to_datetime(0)
        bbox_props = dict(boxstyle='square', facecolor='pink', alpha=0.5)
        self.title_ = self.fig_.text(0.50, 0.90, '%s \n Time: %s' %
                                     (self.pipe_name, elapsed_time),
                                     fontsize=self.font_size_,
                                     ha='center',
                                     bbox=bbox_props)

        # Handle colormap
        cmap_jet = plt.get_cmap('jet')
        clr_norm = colors.Normalize(vmin=0, vmax=1)
        self.scalar_cmap = cmx.ScalarMappable(norm=clr_norm, cmap=cmap_jet)

    def _func_def(self, adj, sec):
        """
        Update the current figure with the adjacency matrix
        """

        # Handle non-significant adjacency values, get upper triangle
        adj[np.isnan(adj)] = 0
        sensor_connect = adj[self.triu_idx_[0], self.triu_idx_[1]]

        self.fig_.canvas.restore_region(self.fig_background_)

        # Iterate over axes and update line objects
        for ax_idx in xrange(self.n_ax_):
            # Restore sensor_map and sensor_nodes (background)
            self.fig_.canvas.restore_region(self.ax_background_[ax_idx])

            for idx, connect in enumerate(self.ax_sensor_connect_[ax_idx]):
                clr_val = self.scalar_cmap.to_rgba(sensor_connect[idx])
                if sensor_connect[idx] > self.adj_lim[ax_idx][0] and \
                        sensor_connect[idx] < self.adj_lim[ax_idx][1]:
                    connect[0].set_color(clr_val)
                    connect[0].set_alpha(0.75)
                else:
                    connect[0].set_alpha(0)
                self.ax_[ax_idx].draw_artist(connect[0])

        # Set new title and update.
        elapsed_time = time_to_datetime(sec)
        self.title_.set_text('%s \n Time: %s' %
                             (self.pipe_name, elapsed_time))
        self.fig_.draw_artist(self.title_)
        self.fig_.canvas.blit(self.title_.get_window_extent())

        # Update all drawn artists in figure
        [self.fig_.canvas.blit(ax.clipbox) for ax in self.ax_]

        self.fig_.canvas.flush_events()
