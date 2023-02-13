import numpy as np
import matplotlib as plt

def vcut(coord, values, left=None, right=None, axis=-1):
    """
    Cut the coordinate and values arrays of a sampled function so as to reduce
    its coordinate range to [`left`, `right`].

    Return views if `copy` is False.
    This do not modify the input arrays.
    None means infinity.
    """
    coord_out = np.asarray(coord)
    values_out = np.swapaxes(values, 0, axis)

    if left is not None:
        left_i = np.searchsorted(coord_out, [left])[0]
        coord_out = coord_out[left_i:]
        values_out = values_out[left_i:]
    if right is not None:
        right_i = np.searchsorted(coord_out, [right])[0]
        if right_i < len(coord_out):
            coord_out = coord_out[:right_i]
            values_out = values_out[:right_i]

    values_out = np.swapaxes(values_out, 0, axis)

    return coord_out, values_out

##### plotting utils #####

def autoscale_y(ax=None, lines=None, margin=0.1, logscale=False):
    """
    This function adapts ylim based on the data that is visible given the current xlim.
    ax -- a matplotlib axes object, current one as default
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    logscale -- indicates if y axis is to be shown in log scale
    """
    if ax is None:
        ax = plt.gca()

    # TODO: remove axvline
    if lines is None:
        lines = ax.get_lines()
    else:
        for line in lines:
            if line not in ax.get_lines():
                raise ValueError("A line provided has not been plotted in `ax`")

    bot, top = np.inf, -np.inf  # we start top < bot
    lo, hi = ax.get_xlim()

    for line in lines:
        xd, yd = line.get_data()
        xd = np.atleast_1d(xd)
        yd = np.atleast_1d(yd)

        # TODO: add error bars
        if len(yd) > 0 and (yd.dtype == float or yd.dtype == int):
            mask = np.logical_and(xd >= lo, xd <= hi)
            mask = np.logical_and(mask, np.isfinite(yd))

            if logscale:
                mask = np.logical_and(mask, yd > 0)

            bot = min(bot, np.min(yd[mask], initial=np.inf))
            top = max(top, np.max(yd[mask], initial=-np.inf))

    if top > bot:
        if logscale:
            h = (top / bot) ** margin
            bot /= h
            top *= h

        else:
            h = (top - bot) * margin
            bot -= h
            top += h
    else:  # bot and top has not been changed by data
        if logscale:
            bot, top = 0.89, 11.22
        else:
            bot, top = -0.055, 0.055

    ax.set_ylim(bot, top)

