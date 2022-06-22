import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.collections as mcol
import matplotlib.transforms as mtransforms

class HexVis():
    @staticmethod
    def hexagonal_imshow(x, y, d, r=1, ax=None, cmap='gray'):
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
                                         transOffset=mtransforms.offset_copy(ax.transData, units='dots'),
                                         array=d,
                                         cmap=cmap,
                                         sizes=size,
                                         linewidths=1,
                                         edgecolors=colormap(d),
                                         )
        col.set_transform(ax.transData)
        ax.add_collection(col)
        ax.set_xlim(np.min(x), np.max(x))
        ax.set_ylim(np.min(y), np.max(y))
        ax.set_aspect(1)
        if ax is None:
            fig.show()