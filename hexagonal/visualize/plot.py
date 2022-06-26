import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as mcol
import matplotlib.transforms as mtransforms
from functools import cached_property


class hexagonal_imshow():
    def __init__(self, x, y, d, r=1, ax=None, cmap='gray'):
        self.x = x
        self.y = y

        # Check ax is given or not.
        self.if_ax(ax)

        # Get colormap
        colormap = mpl.cm.get_cmap(cmap)

        d = d / np.max(d)  # Normalizing data range (0 - 1)
        vertices = r
        area = vertices ** 2
        size = np.ones(shape=np.shape(x)) * area

        # Transform data locations.

        self.col = mcol.RegularPolyCollection(6,
                                              offsets=list(zip(self.__transformed_xyz_matrix[0], self.__transformed_xyz_matrix[1])),
                                              array=d,
                                              cmap=cmap,
                                              sizes=size,
                                              linewidths=None,
                                              edgecolors=None,
                                              )
        self.ax.add_collection(self.col)
        self.fig.canvas.mpl_connect('draw_event', self.on_draw)
        self.fig.canvas.draw()

    def if_ax(self, ax:plt.Axes=None):
        if ax is None:
            self.fig, self.ax = plt.subplots()

        else:
            self.ax = ax
            self.fig = ax.get_figure()

    def on_draw(self, event):
        transdata_sanitized = self.ax.transData.get_matrix()
        transdata_sanitized[0, 2] = 0
        transdata_sanitized[1, 2] = 0
        transdata_sanitized[0, 1] = 0
        transdata_sanitized[1, 0] = 0

        if isinstance(self.ax.get_transform(), mtransforms.IdentityTransform):
            self.col.set_transform(mtransforms.Affine2D(transdata_sanitized))
        else:
            ax_transform_translation_sanitized = self.ax.get_transform()
            ax_transform_translation_sanitized[0, 2] = 0
            ax_transform_translation_sanitized[1, 2] = 0
            ax_transform_translation_sanitized[0, 0] = 1
            ax_transform_translation_sanitized[1, 1] = 1
            self.col.set_transform(mtransforms.Affine2D(ax_transform_translation_sanitized @ transdata_sanitized))
        self.col.set_offset_transform(self.ax.transData)

    @cached_property
    def __data_xyz_matrix(self):
        """
        Internal usage only. Returns array comprised with [x, y, 1] matrix. Last 1 was added to transfer x, y coordinate
         to the homogeneous affine transformation space.
        :return:
        """
        return list(zip(self.x, self.y, np.ones(np.shape(self.x))))

    @cached_property
    def __transformed_xyz_matrix(self):
        """
        Internal usage only. Returns array which contains transformed x, y coordinate. Please note, this code only
        aware affine transformation that provided by `set_transform()`.
        """
        if isinstance(self.ax.get_transform(), mtransforms.IdentityTransform):
            return np.array(self.__data_xyz_matrix).transpose()
        else:
            return self.ax.get_transform() @ np.array(self.__data_xyz_matrix).transpose()

    @cached_property
    def ax(self) -> plt.Axes:
        """
        Return the Axes object.
        """
        return self.ax

    @cached_property
    def fig(self):
        """
        Return the figure object.
        """
        return self.fig

    @cached_property
    def extent(self):
        """
        Return the extent of the original data. If transformed data extent required please use `deformed_extent`
        attribute.
        """
        return np.min(self.x), \
               np.max(self.x), \
               np.min(self.y), \
               np.max(self.y)

    @cached_property
    def deformed_extent(self):
        """
        Return the extent of the deformed data. If original data extent required please use `extent` attribute.
        """
        return np.min(self.__transformed_xyz_matrix[0]), \
               np.max(self.__transformed_xyz_matrix[0]), \
               np.min(self.__transformed_xyz_matrix[1]), \
               np.max(self.__transformed_xyz_matrix[1])
