"""Plotting functions."""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.ticker import LogLocator
from matplotlib.ticker import MaxNLocator

from scipy.optimize import OptimizeResult

from skopt.space import Categorical
from collections import Counter


def plot_convergence(*args, **kwargs):
    """Plot one or several convergence traces.

    Parameters
    ----------
    * `args[i]` [`OptimizeResult`, list of `OptimizeResult`, or tuple]:
        The result(s) for which to plot the convergence trace.

        - if `OptimizeResult`, then draw the corresponding single trace;
        - if list of `OptimizeResult`, then draw the corresponding convergence
          traces in transparency, along with the average convergence trace;
        - if tuple, then `args[i][0]` should be a string label and `args[i][1]`
          an `OptimizeResult` or a list of `OptimizeResult`.

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `true_minimum` [float, optional]:
        The true minimum value of the function, if known.

    * `yscale` [None or string, optional]:
        The scale for the y-axis.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """

    # TODO: Please document the code in this function.

    # <3 legacy python
    ax = kwargs.get("ax", None)
    true_minimum = kwargs.get("true_minimum", None)
    yscale = kwargs.get("yscale", None)

    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel("Number of calls $n$")
    ax.set_ylabel(r"$\min f(x)$ after $n$ calls")
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    colors = cm.viridis(np.linspace(0.25, 1.0, len(args)))

    for results, color in zip(args, colors):
        if isinstance(results, tuple):
            name, results = results
        else:
            name = None

        if isinstance(results, OptimizeResult):
            n_calls = len(results.x_iters)
            mins = [np.min(results.func_vals[:i])
                    for i in range(1, n_calls + 1)]
            ax.plot(range(1, n_calls + 1), mins, c=color,
                    marker=".", markersize=12, lw=2, label=name)

        elif isinstance(results, list):
            n_calls = len(results[0].x_iters)
            iterations = range(1, n_calls + 1)
            mins = [[np.min(r.func_vals[:i]) for i in iterations]
                    for r in results]

            for m in mins:
                ax.plot(iterations, m, c=color, alpha=0.2)

            ax.plot(iterations, np.mean(mins, axis=0), c=color,
                    marker=".", markersize=12, lw=2, label=name)

    if true_minimum:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum or name:
        ax.legend(loc="best")

    return ax

# TODO: Please write doc-strings and document the code in this function.
def _format_scatter_plot_axes(ax, space, ylabel, dim_labels=None):
    # Work out min, max of y axis for the diagonal so we can adjust
    # them all to the same value
    diagonal_ylim = (np.min([ax[i, i].get_ylim()[0]
                             for i in range(space.n_dims)]),
                     np.max([ax[i, i].get_ylim()[1]
                             for i in range(space.n_dims)]))

    # TODO: The above is very confusing code-style because
    # TODO: of the deep nesting and list-comprehension.
    # TODO: Splitting it into several lines makes it easier to read and debug.
    # # Get ylim for all diagonal plots.
    # ylim = [ax[i, i].get_ylim() for i in range(n_dims)]
    #
    # # Separate into two lists with low and high ylim.
    # ylim_lo, ylim_hi = zip(*ylim)
    #
    # # Min ylim for all diagonal plots.
    # ylim_min = np.min(ylim_lo)
    #
    # # Max ylim for all diagonal plots.
    # ylim_max = np.max(ylim_hi)
    #
    # # Tuple for use with set_ylim() below.
    # diagonal_ylim = (ylim_min, ylim_max)

    # TODO: This should call a function in the space-object.
    # TODO: Perhaps rewrite to use the new space.get_dimensions() ?
    if dim_labels is None:
        dim_labels = ["$X_{%i}$" % i if d.name is None else d.name
                for i, d in enumerate(space.dimensions)]

    # TODO: The nested for-loops below are used in several of the functions.
    # TODO: The nesting-depth is too high here. The following structure is better:
    # for i in range(n_dims):
    #     # Do something for the diagonal case here.
    #
    #     for j in range(i):
    #         # Do something for the case where j<i.
    #
    #     for j in range(i+1, n_dims):
    #         # Do something for the case where j>i.

    # Deal with formatting of the axes
    for i in range(space.n_dims):  # rows
        for j in range(space.n_dims):  # columns
            ax_ = ax[i, j]

            if j > i:
                ax_.axis("off")

            # off-diagonal axis
            if i != j:
                # TODO: This comment is not funny, it is just confusing!
                # plots on the diagonal are special, like Texas. They have
                # their own range so do not mess with them.
                ax_.set_ylim(*space.dimensions[i].bounds)
                ax_.set_xlim(*space.dimensions[j].bounds)
                if j > 0:
                    ax_.set_yticklabels([])
                else:
                    ax_.set_ylabel(dim_labels[i])

                # for all rows except ...
                if i < space.n_dims - 1:
                    ax_.set_xticklabels([])
                # ... the bottom row
                else:
                    [l.set_rotation(45) for l in ax_.get_xticklabels()]
                    ax_.set_xlabel(dim_labels[j])

                # TODO: An English comment starts with a capital letter and ends with a .
                # configure plot for linear vs log-scale
                priors = (space.dimensions[j].prior, space.dimensions[i].prior)
                scale_setters = (ax_.set_xscale, ax_.set_yscale)
                loc_setters = (ax_.xaxis.set_major_locator,
                               ax_.yaxis.set_major_locator)
                for set_major_locator, set_scale, prior in zip(
                        loc_setters, scale_setters, priors):
                    if prior == 'log-uniform':
                        set_scale('log')
                    else:
                        set_major_locator(MaxNLocator(6, prune='both'))

            else:
                ax_.set_ylim(*diagonal_ylim)
                ax_.yaxis.tick_right()
                ax_.yaxis.set_label_position('right')
                ax_.yaxis.set_ticks_position('both')
                ax_.set_ylabel(ylabel)

                ax_.xaxis.tick_top()
                ax_.xaxis.set_label_position('top')
                ax_.set_xlabel(dim_labels[j])

                if space.dimensions[i].prior == 'log-uniform':
                    ax_.set_xscale('log')
                else:
                    ax_.xaxis.set_major_locator(MaxNLocator(6, prune='both'))

    return ax


def partial_dependence(space, model, i, j=None, sample_points=None,
                       n_samples=250, n_points=40):
    """Calculate the partial dependence for dimensions `i` and `j` with
    respect to the objective value, as approximated by `model`.

    The partial dependence plot shows how the value of the dimensions
    `i` and `j` influence the `model` predictions after "averaging out"
    the influence of all other dimensions.

    Parameters
    ----------
    * `space` [`Space`]
        The parameter space over which the minimization was performed.

    * `model`
        Surrogate model for the objective function.

    * `i` [int]
        The first dimension for which to calculate the partial dependence.

    * `j` [int, default=None]
        The second dimension for which to calculate the partial dependence.
        To calculate the 1D partial dependence on `i` alone set `j=None`.

    * `sample_points` [np.array, shape=(n_points, n_dims), default=None]
        Randomly sampled and transformed points to use when averaging
        the model function at each of the `n_points`.

    * `n_samples` [int, default=100]
        Number of random samples to use for averaging the model function
        at each of the `n_points`. Only used when `sample_points=None`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension `i` and `j`.

    Returns
    -------
    For 1D partial dependence:

    * `xi`: [np.array]:
        The points at which the partial dependence was evaluated.

    * `yi`: [np.array]:
        The value of the model at each point `xi`.

    For 2D partial dependence:

    * `xi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `yi`: [np.array, shape=n_points]:
        The points at which the partial dependence was evaluated.
    * `zi`: [np.array, shape=(n_points, n_points)]:
        The value of the model at each point `(xi, yi)`.
    """

    # TODO: Please comment this code. I have no idea what it does!

    if sample_points is None:
        sample_points = space.transform(space.rvs(n_samples=n_samples))

    if j is None:
        bounds = space.dimensions[i].bounds
        # XXX use linspace(*bounds, n_points) after python2 support ends
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = space.dimensions[i].transform(xi)

        yi = []
        for x_ in xi_transformed:
            rvs_ = np.array(sample_points)
            rvs_[:, i] = x_
            yi.append(np.mean(model.predict(rvs_)))

        return xi, yi

    else:
        # XXX use linspace(*bounds, n_points) after python2 support ends
        bounds = space.dimensions[j].bounds
        xi = np.linspace(bounds[0], bounds[1], n_points)
        xi_transformed = space.dimensions[j].transform(xi)

        bounds = space.dimensions[i].bounds
        yi = np.linspace(bounds[0], bounds[1], n_points)
        yi_transformed = space.dimensions[i].transform(yi)

        zi = []
        for x_ in xi_transformed:
            row = []
            for y_ in yi_transformed:
                rvs_ = np.array(sample_points)
                rvs_[:, (j, i)] = (x_, y_)
                row.append(np.mean(model.predict(rvs_)))
            zi.append(row)

        return xi, yi, np.array(zi).T


# TODO: The 'dimensions' arg is strange. Why not use something like a list of
# TODO: dimension_ids similar to the `plot_contour()` function below?
def plot_objective(result, levels=10, n_points=40, n_samples=250, size=2,
                   zscale='linear', dimensions=None):
    # TODO: I don't know what "pairwise partial dependence" is.
    # TODO: Could you explain this in English?
    # TODO: Please also explain the diagonal plots better. How do I interpret them?
    """Pairwise partial dependence plot of the objective function.

    The diagonal shows the partial dependence for dimension `i` with
    respect to the objective function. The off-diagonal shows the
    partial dependence for dimensions `i` and `j` with
    respect to the objective function. The objective function is
    approximated by `result.model.`

    Pairwise scatter plots of the points at which the objective
    function was directly evaluated are shown on the off-diagonal.
    A red point indicates the found minimum.

    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot, passed directly
        to `plt.contour()`.

    * `n_points` [int, default=40]
        Number of points at which to evaluate the partial dependence
        along each dimension.

    * `n_samples` [int, default=250]
        Number of random samples to use for averaging the model function
        at each of the `n_points`.

    * `size` [float, default=2]
        Height (in inches) of each facet.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'linear'
        or 'log'.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    space = result.space

    # TODO: This is incorrect when the search-space has dimensions
    # TODO: with different types. Please use _get_samples_dimension() instead.
    samples = np.asarray(result.x_iters)

    rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

    # TODO: Make another convenience variable: n_dims = space.n_dims

    # TODO: Use space.get_dimensions(ids=dimension_ids) to get the
    # TODO: relevant dimensions and their indices and names. This
    # TODO: would allow us to call this function with search-spaces
    # TODO: containing categorical dimensions by simply omitting them.

    # TODO: Make a check that the list of dimensions does not contain
    # TODO: categorical ones, so the user knows that is unsupported.

    if zscale == 'log':
        locator = LogLocator()
    elif zscale == 'linear':
        locator = None
    else:
        raise ValueError("Valid values for zscale are 'linear' and 'log',"
                         " not '%s'." % zscale)

    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(size * space.n_dims, size * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    # TODO: See comments above about the nested for-loops.
    # TODO: And please comment this code.
    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                xi, yi = partial_dependence(space, result.models[-1], i,
                                            j=None,
                                            sample_points=rvs_transformed,
                                            n_points=n_points)

                ax[i, i].plot(xi, yi)
                ax[i, i].axvline(result.x[i], linestyle="--", color="r", lw=1)

            # lower triangle
            elif i > j:
                xi, yi, zi = partial_dependence(space, result.models[-1],
                                                i, j,
                                                rvs_transformed, n_points)
                ax[i, j].contourf(xi, yi, zi, levels,
                                  locator=locator, cmap='viridis_r')
                ax[i, j].scatter(samples[:, j], samples[:, i],
                                 c='k', s=10, lw=0.)
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    # TODO: Why not return the fig-object? It is easier to save as a file.
    return _format_scatter_plot_axes(ax, space, ylabel="Partial dependence",
                                     dim_labels=dimensions)

# TODO: The same comment as for plot_objective() above where we should take
# TODO: an arg dimension_ids so we can call this function with the dimensions
# TODO: we want to plot. This would make it work for categorical search-spaces.
def plot_evaluations(result, bins=20, dimensions=None):
    """Visualize the order in which points where sampled.

    The scatter plot matrix shows at which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The result for which to create the scatter plot matrix.

    * `bins` [int, bins=20]:
        Number of bins to use for histograms on the diagonal.

    * `dimensions` [list of str, default=None] Labels of the dimension
        variables. `None` defaults to `space.dimensions[i].name`, or
        if also `None` to `['X_0', 'X_1', ..]`.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """

    # TODO: Same comments as for plot_objective() above.
    # TODO: Please comment this code and clean up the for-loops nesting.

    space = result.space

    # TODO: This is incorrect when the search-space has dimensions
    # TODO: with different types. Please use _get_samples_dimension() instead.
    samples = np.asarray(result.x_iters)
    order = range(samples.shape[0])

    fig, ax = plt.subplots(space.n_dims, space.n_dims,
                           figsize=(2 * space.n_dims, 2 * space.n_dims))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(space.n_dims):
        for j in range(space.n_dims):
            if i == j:
                if space.dimensions[j].prior == 'log-uniform':
                    low, high = space.bounds[j]
                    bins_ = np.logspace(np.log10(low), np.log10(high), bins)
                else:
                    bins_ = bins
                ax[i, i].hist(samples[:, j], bins=bins_,
                              range=space.dimensions[j].bounds)

            # lower triangle
            elif i > j:
                ax[i, j].scatter(samples[:, j], samples[:, i], c=order,
                                 s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(result.x[j], result.x[i],
                                 c=['r'], s=20, lw=0.)

    # TODO: Why not return the fig-object? It is easier to save as a file.
    return _format_scatter_plot_axes(ax, space, ylabel="Number of samples",
                                     dim_labels=dimensions)


# TODO: I added this function.
# TODO: Maybe not the best name? Maybe it belongs in utils.py?
def _get_samples_dimension(result, index):
    """Get the samples for the given dimension index
    from the optimization-result from e.g. `gp_minimize()`.

    This function is used instead of numpy, because if
    we convert `result.x_iters` to a 2-d numpy array,
    then all data-types must be identical otherwise numpy
    will promote all the types to the most general type.
    For example, if you have a Categorical dimension which
    is a string, then your Real and Integer dimensions will
    be converted to strings as well in the 2-d numpy array.

    Using this function instead of numpy ensures the
    original data-type is being preserved.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The optimization results e.g. from calling `gp_minimize()`.

    * `index` [int]:
        Index for a dimension in the search-space.

    Returns
    -------
    * `samples`: [list of either int, float or string]:
        The optimization samples for the given dimension.
    """

    # Get the samples from the optimization-log for the relevant dimension.
    samples = [x[index] for x in result.x_iters]

    return samples

def plot_histogram(result, dimension_id, bins=20, rotate_label=False):
    """Create and return a Matplotlib figure with a histogram
    of the samples from the optimization results,
    for a given dimension of the search-space.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The optimization results e.g. from calling `gp_minimize()`.

    * `dimension_id` [int or str]:
        Either an index or name for a dimension in the search-space.

    * `bins` [int, bins=20]:
        Number of bins in the histogram.

    * `rotate_label` [bool, rotate_label=False]:
        Whether or not to rotate the category-names on the x-axis.

    Returns
    -------
    * `fig`: [`matplotlib.figure.Figure`]:
        The Matplotlib Figure-object.
        For example, you can save the plot by calling `fig.savefig('file.png')`
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the dimension-object, its index in the search-space, and its name.
    dimension, index, dimension_name = space.get_dimension(id=dimension_id)

    # Get the samples from the optimization-log for that particular dimension.
    samples = _get_samples_dimension(result=result, index=index)

    # Start a new plot.
    fig = plt.figure()

    if type(dimension) == Categorical:
        # When the search-space dimension is Categorical, it means
        # that the possible values are strings. Matplotlib's histogram
        # does not support this, so we have to make a bar-plot instead.

        # NOTE: This only shows the categories that are in the samples.
        # So if a category was not sampled, it will not be shown here.

        # Count the number of occurrences of the string-categories.
        counter = Counter(samples)

        # The counter returns a dict where the keys are the category-names
        # and the values are the number of occurrences for each category.
        names = list(counter.keys())
        counts = list(counter.values())

        # Although Matplotlib's docs indicate that the bar() function
        # can take a list of strings for the x-axis, it doesn't appear to work.
        # So we hack it by creating a list of integers and setting the
        # tick-labels with the category-names instead.
        x = np.arange(len(counts))

        # Plot using bars.
        plt.bar(x, counts, tick_label=names)

        # Rotate the category-names 90 degrees.
        if rotate_label:
            plt.xticks(rotation=90)
    else:
        # When the search-space Dimension is either integer or float,
        # the histogram can be plotted directly.
        plt.hist(samples, bins=bins, range=dimension.bounds)

    # Set the labels.
    plt.xlabel(dimension_name)
    plt.ylabel('Number of samples')

    return fig


def plot_contour(result, dimension_id1, dimension_id2,
                 n_points=40, n_samples=250, levels=10, zscale='linear'):
    """Create and return a Matplotlib figure with a landscape
    contour-plot of the last fitted model of the search-space,
    overlaid with all the samples from the optimization results,
    for the two given dimensions of the search-space.

    Parameters
    ----------
    * `result` [`OptimizeResult`]
        The optimization results e.g. from calling `gp_minimize()`.

    * `dimension_id1` [int or str]:
        Either an index or name for a dimension in the search-space.

    * `dimension_id2` [int or str]:
        Either an index or name for a dimension in the search-space.

    * `n_samples` [int, default=250]
        Number of random samples used for estimating the contour-plot
        of the objective function.

    * `n_points` [int, default=40]
        Number of points along each dimension where the partial dependence
        is evaluated.

    * `levels` [int, default=10]
        Number of levels to draw on the contour plot.

    * `zscale` [str, default='linear']
        Scale to use for the z axis of the contour plots. Either 'log'
        or linear for all other choices.

    Returns
    -------
    * `fig`: [`matplotlib.figure.Figure`]:
        The Matplotlib Figure-object.
        For example, you can save the plot by calling `fig.savefig('file.png')` 
    """

    # Get the search-space instance from the optimization results.
    space = result.space

    # Get the dimension-object, its index in the search-space, and its name.
    dimension1, index1, dimension_name1 = space.get_dimension(id=dimension_id1)
    dimension2, index2, dimension_name2 = space.get_dimension(id=dimension_id2)

    # Ensure dimensions are not Categorical.
    if type(dimension1) == Categorical or type(dimension2) == Categorical:
        raise ValueError("Categorical dimension is not supported.")

    # Get the samples from the optimization-log for the relevant dimensions.
    samples1 = _get_samples_dimension(result=result, index=index1)
    samples2 = _get_samples_dimension(result=result, index=index2)

    # Get the best-found samples for the relevant dimensions.
    best_sample1 = result.x[index1]
    best_sample2 = result.x[index2]

    # Get the last fitted model for the search-space.
    last_model = result.models[-1]

    # Get new random samples from the search-space and transform if necessary.
    sample_points = space.rvs(n_samples=n_samples)
    sample_points = space.transform(sample_points)

    # Estimate the objective function for these sampled points
    # using the last fitted model for the search-space.
    xi, yi, zi = partial_dependence(space=space,
                                    model=last_model,
                                    i=index2, j=index1,
                                    sample_points=sample_points,
                                    n_points=n_points)

    # Start a new plot.
    fig = plt.figure()

    # Scale for the z-axis of the contour-plot. Either Log or Linear (None).
    locator = LogLocator() if zscale == 'log' else None

    # Use the min and max values of the objective function from the
    # optimization results to normalize the color-gradient across plots for
    # different choices of dimensions.
    # TODO: I'm not sure if this should be used?
    vmin = np.min(result.func_vals)
    vmax = np.max(result.func_vals)

    # Plot the contour-landscape for the objective function.
    plt.contourf(xi, yi, zi, levels, locator=locator, cmap='viridis_r',
                 vmin=vmin, vmax=vmax)

    # Plot all the parameters that were sampled during optimization.
    # These are plotted as small black dots.
    plt.scatter(samples1, samples2, c='black', s=10, linewidths=1)

    # Plot the best parameters that were sampled during optimization.
    # These are plotted as a big red star.
    plt.scatter(best_sample1, best_sample2,
                c='red', s=50, linewidths=1, marker='*')

    # Use the dimension-names as the labels for the plot-axes.
    plt.xlabel(dimension_name1)
    plt.ylabel(dimension_name2)

    return fig
