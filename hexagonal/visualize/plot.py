import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as mcol
import matplotlib.transforms as mtransforms

class HexVis():
    @staticmethod
    def hexagonal_imshow(x, y, d, r=1, ax=None, cmap='gray'):
        # Get colormap.
        colormap = mpl.cm.get_cmap(cmap)

        if ax is not None:
            ax = ax
        else:
            fig, ax = plt.subplots()

        d = d / np.max(d)
        vertices = r
        area = vertices ** 2
        size = np.ones(shape=np.shape(x)) * area
        col = mcol.RegularPolyCollection(6,
                                         offsets=list(zip(x, y)),
                                         array=d,
                                         cmap=cmap,
                                         sizes=size,
                                         linewidths=1,
                                         edgecolors=colormap(d),
                                         )
        polygon_offset_transformation = ax.transData + mtransforms.Affine2D(ax.get_transform())
        col.set_offset_transform(polygon_offset_transformation)
        col.set_transform(ax.transData + mtransforms.Affine2D(ax.get_transform()))
        ax.add_collection(col)
        ax.autoscale()
