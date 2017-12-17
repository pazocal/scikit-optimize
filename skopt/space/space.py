import numbers
import numpy as np

from scipy.stats.distributions import randint
from scipy.stats.distributions import rv_discrete
from scipy.stats.distributions import uniform

from sklearn.utils import check_random_state
from sklearn.utils.fixes import sp_version

from .transformers import CategoricalEncoder
from .transformers import Normalize
from .transformers import Identity
from .transformers import Log10
from .transformers import Pipeline


# helper class to be able to print [1, ..., 4] instead of [1, '...', 4]
class _Ellipsis:
    def __repr__(self):
        return '...'


def check_dimension(dimension, transform=None):
    """Turn a provided dimension description into a dimension object.

    Checks that the provided dimension falls into one of the
    supported types. For a list of supported types, look at
    the documentation of ``dimension`` below.

    If ``dimension`` is already a ``Dimension`` instance, return it.

    Parameters
    ----------
    * `dimension`:
        Search space Dimension.
        Each search dimension can be defined either as

        - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
          dimensions),
        - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
          dimensions),
        - as a list of categories (for `Categorical` dimensions), or
        - an instance of a `Dimension` object (`Real`, `Integer` or
          `Categorical`).

    * `transform` ["identity", "normalize", "onehot" optional]:
        - For `Categorical` dimensions, the following transformations are
          supported.

          - "onehot" (default) one-hot transformation of the original space.
          - "identity" same as the original space.

        - For `Real` and `Integer` dimensions, the following transformations
          are supported.

          - "identity", (default) the transformed space is the same as the
            original space.
          - "normalize", the transformed space is scaled to be between 0 and 1.

    Returns
    -------
    * `dimension`:
        Dimension instance.
    """
    if isinstance(dimension, Dimension):
        return dimension

    if not isinstance(dimension, (list, tuple, np.ndarray)):
        raise ValueError("Dimension has to be a list or tuple.")

    if len(dimension) == 2:
        if any([isinstance(d, (str, bool)) for d in dimension]):
            return Categorical(dimension, transform=transform)
        elif all([isinstance(dim, numbers.Integral) for dim in dimension]):
            return Integer(*dimension, transform=transform)
        elif any([isinstance(dim, numbers.Real) for dim in dimension]):
            return Real(*dimension, transform=transform)
        else:
            raise ValueError("Invalid dimension {}. Read the documentation for"
                             " supported types.".format(dimension))

    if len(dimension) == 3:
        if (any([isinstance(dim, (float, int)) for dim in dimension[:2]]) and
            dimension[2] in ["uniform", "log-uniform"]):
            return Real(*dimension, transform=transform)
        else:
            return Categorical(dimension, transform=transform)

    if len(dimension) > 3:
        return Categorical(dimension, transform=transform)

    raise ValueError("Invalid dimension {}. Read the documentation for "
                     "supported types.".format(dimension))


class Dimension(object):
    """Base class for search space dimensions."""

    prior = None

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        Parameters
        ----------
        * `n_samples` [int or None]:
            The number of samples to be drawn.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.
        """
        rng = check_random_state(random_state)
        samples = self._rvs.rvs(size=n_samples, random_state=rng)
        return self.inverse_transform(samples)

    def transform(self, X):
        """Transform samples form the original space to a warped space."""
        return self.transformer.transform(X)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           original space.
        """
        return self.transformer.inverse_transform(Xt)

    @property
    def size(self):
        return 1

    @property
    def transformed_size(self):
        return 1

    @property
    def bounds(self):
        raise NotImplementedError

    @property
    def transformed_bounds(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        if isinstance(value, str) or value is None:
            self._name = value
        else:
            raise ValueError("Dimension's name must be either string or None.")


def _uniform_inclusive(loc=0.0, scale=1.0):
    # like scipy.stats.distributions but inclusive of `high`
    # XXX scale + 1. might not actually be a float after scale if
    # XXX scale is very large.
    return uniform(loc=loc, scale=np.nextafter(scale, scale + 1.))


class Real(Dimension):
    def __init__(self, low, high, prior="uniform", transform=None, name=None):
        """Search space dimension that can take on any real value.

        Parameters
        ----------
        * `low` [float]:
            Lower bound (inclusive).

        * `high` [float]:
            Upper bound (inclusive).

        * `prior` ["uniform" or "log-uniform", default="uniform"]:
            Distribution to use when sampling random points for this dimension.
            - If `"uniform"`, points are sampled uniformly between the lower
              and upper bounds.
            - If `"log-uniform"`, points are sampled uniformly between
              `log10(lower)` and `log10(upper)`.`

        * `transform` ["identity", "normalize", optional]:
            The following transformations are supported.

            - "identity", (default) the transformed space is the same as the
              original space.
            - "normalize", the transformed space is scaled to be between
              0 and 1.

        * `name` [str or None]:
            Name associated with the dimension, e.g., "learning rate".
        """
        if high <= low:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(low, high))
        self.low = low
        self.high = high
        self.prior = prior
        self.name = name

        if transform is None:
            transform = "identity"

        self.transform_ = transform

        if self.transform_ not in ["normalize", "identity"]:
            raise ValueError("transform should be 'normalize' or 'identity'"
                             " got {}".format(self.transform_))

        # Define _rvs and transformer spaces.
        # XXX: The _rvs is for sampling in the transformed space.
        # The rvs on Dimension calls inverse_transform on the points sampled
        # using _rvs
        if self.transform_ == "normalize":
            # set upper bound to next float after 1. to make the numbers
            # inclusive of upper edge
            self._rvs = _uniform_inclusive(0., 1.)
            if self.prior == "uniform":
                self.transformer = Pipeline(
                    [Identity(), Normalize(low, high)])
            else:
                self.transformer = Pipeline(
                    [Log10(), Normalize(np.log10(low), np.log10(high))]
                )
        else:
            if self.prior == "uniform":
                self._rvs = _uniform_inclusive(self.low, self.high - self.low)
                self.transformer = Identity()
            else:
                self._rvs = _uniform_inclusive(
                    np.log10(self.low),
                    np.log10(self.high) - np.log10(self.low))
                self.transformer = Log10()

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.allclose([self.low], [other.low]) and
                np.allclose([self.high], [other.high]) and
                self.prior == other.prior and
                self.transform_ == other.transform_)

    def __repr__(self):
        return "Real(low={}, high={}, prior={}, transform={}, name={})".format(
            self.low, self.high, self.prior, self.transform_, self.name)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        return super(Real, self).inverse_transform(Xt).astype(np.float)

    @property
    def bounds(self):
        return (self.low, self.high)

    def __contains__(self, point):
        return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0.0, 1.0
        else:
            if self.prior == "uniform":
                return self.low, self.high
            else:
                return np.log10(self.low), np.log10(self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        * `a` [float]
            First point.

        * `b` [float]
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError("Can only compute distance for values within "
                               "the space, not %s and %s." % (a, b))
        return abs(a - b)


class Integer(Dimension):
    def __init__(self, low, high, transform=None, name=None):
        """Search space dimension that can take on integer values.

        Parameters
        ----------
        * `low` [int]:
            Lower bound (inclusive).

        * `high` [int]:
            Upper bound (inclusive).

        * `transform` ["identity", "normalize", optional]:
            The following transformations are supported.

            - "identity", (default) the transformed space is the same as the
              original space.
            - "normalize", the transformed space is scaled to be between
              0 and 1.

        * `name` [str or None]:
            Name associated with dimension, e.g., "number of trees".
        """
        if high <= low:
            raise ValueError("the lower bound {} has to be less than the"
                             " upper bound {}".format(low, high))
        self.low = low
        self.high = high
        self.name = name

        if transform is None:
            transform = "identity"

        self.transform_ = transform

        if transform not in ["normalize", "identity"]:
            raise ValueError("transform should be 'normalize' or 'identity'"
                             " got {}".format(self.transform_))
        if transform == "normalize":
            self._rvs = uniform(0, 1)
            self.transformer = Normalize(low, high, is_int=True)
        else:
            self._rvs = randint(self.low, self.high + 1)
            self.transformer = Identity()

    def __eq__(self, other):
        return (type(self) is type(other) and
                np.allclose([self.low], [other.low]) and
                np.allclose([self.high], [other.high]))

    def __repr__(self):
        return "Integer(low={}, high={}, name={})".format(self.low, self.high, self.name)

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back into the
           orignal space.
        """
        # The concatenation of all transformed dimensions makes Xt to be
        # of type float, hence the required cast back to int.
        return super(Integer, self).inverse_transform(Xt).astype(np.int)

    @property
    def bounds(self):
        return (self.low, self.high)

    def __contains__(self, point):
        return self.low <= point <= self.high

    @property
    def transformed_bounds(self):
        if self.transform_ == "normalize":
            return 0, 1
        else:
            return (self.low, self.high)

    def distance(self, a, b):
        """Compute distance between point `a` and `b`.

        Parameters
        ----------
        * `a` [int]
            First point.

        * `b` [int]
            Second point.
        """
        if not (a in self and b in self):
            raise RuntimeError("Can only compute distance for values within "
                               "the space, not %s and %s." % (a, b))
        return abs(a - b)


class Categorical(Dimension):
    def __init__(self, categories, prior=None, transform=None, name=None):
        """Search space dimension that can take on categorical values.

        Parameters
        ----------
        * `categories` [list, shape=(n_categories,)]:
            Sequence of possible categories.

        * `prior` [list, shape=(categories,), default=None]:
            Prior probabilities for each category. By default all categories
            are equally likely.

        * `transform` ["onehot", "identity", default="onehot"] :
            - "identity", the transformed space is the same as the original
              space.
            - "onehot", the transformed space is a one-hot encoded
              representation of the original space.

        * `name` [str or None]:
            Name associated with dimension, e.g., "colors".
        """
        self.categories = tuple(categories)
        self.name = name

        if transform is None:
            transform = "onehot"
        self.transform_ = transform
        if transform not in ["identity", "onehot"]:
            raise ValueError("Expected transform to be 'identity' or 'onehot' "
                             "got {}".format(transform))
        if transform == "onehot":
            self.transformer = CategoricalEncoder()
            self.transformer.fit(self.categories)
        else:
            self.transformer = Identity()
        self.prior = prior

        if prior is None:
            self.prior_ = np.tile(1. / len(self.categories),
                                  len(self.categories))
        else:
            self.prior_ = prior

        # XXX check that sum(prior) == 1
        self._rvs = rv_discrete(
            values=(range(len(self.categories)), self.prior_)
            )

    def __eq__(self, other):
        return (type(self) is type(other) and
                self.categories == other.categories and
                np.allclose(self.prior_, other.prior_))

    def __repr__(self):
        if len(self.categories) > 7:
            cats = self.categories[:3] + [_Ellipsis()] + self.categories[-3:]
        else:
            cats = self.categories

        if self.prior is not None and len(self.prior) > 7:
            prior = self.prior[:3] + [_Ellipsis()] + self.prior[-3:]
        else:
            prior = self.prior

        return "Categorical(categories={}, prior={}, name={})".format(cats, prior, self.name)

    def rvs(self, n_samples=None, random_state=None):
        choices = self._rvs.rvs(size=n_samples, random_state=random_state)

        if isinstance(choices, numbers.Integral):
            return self.categories[choices]
        else:
            return [self.categories[c] for c in choices]

    @property
    def transformed_size(self):
        if self.transform_ == "onehot":
            size = len(self.categories)
            # when len(categories) == 2, CategoricalEncoder outputs a
            # single value
            return size if size != 2 else 1
        return 1

    @property
    def bounds(self):
        return self.categories

    def __contains__(self, point):
        return point in self.categories

    @property
    def transformed_bounds(self):
        if self.transformed_size == 1:
            return (0.0, 1.0)
        else:
            return [(0.0, 1.0) for i in range(self.transformed_size)]

    def distance(self, a, b):
        """Compute distance between category `a` and `b`.

        As categories have no order the distance between two points is one
        if a != b and zero otherwise.

        Parameters
        ----------
        * `a` [category]
            First category.

        * `b` [category]
            Second category.
        """
        if not (a in self and b in self):
            raise RuntimeError("Can only compute distance for values within"
                               " the space, not {} and {}.".format(a, b))
        return 1 if a != b else 0


class Space(object):
    """Search space."""

    def __init__(self, dimensions):
        """Initialize a search space from given specifications.

        Parameters
        ----------
        * `dimensions` [list, shape=(n_dims,)]:
            List of search space dimensions.
            Each search dimension can be defined either as

            - a `(lower_bound, upper_bound)` tuple (for `Real` or `Integer`
              dimensions),
            - a `(lower_bound, upper_bound, "prior")` tuple (for `Real`
              dimensions),
            - as a list of categories (for `Categorical` dimensions), or
            - an instance of a `Dimension` object (`Real`, `Integer` or
              `Categorical`).

            NOTE: The upper and lower bounds are inclusive for `Integer`
            dimensions.
        """
        self.dimensions = [check_dimension(dim) for dim in dimensions]

        # Names of all the dimensions in the search-space.
        self.dimension_names = [dim.name for dim in self.dimensions]

    def __eq__(self, other):
        return all([a == b for a, b in zip(self.dimensions, other.dimensions)])

    def __repr__(self):
        if len(self.dimensions) > 31:
            dims = self.dimensions[:15] + [_Ellipsis()] + self.dimensions[-15:]
        else:
            dims = self.dimensions
        return "Space([{}])".format(',\n       '.join(map(str, dims)))

    def __iter__(self):
        return iter(self.dimensions)

    @property
    def is_real(self):
        """
        Returns true if all dimensions are Real
        """
        return all([isinstance(dim, Real) for dim in self.dimensions])

    def rvs(self, n_samples=1, random_state=None):
        """Draw random samples.

        The samples are in the original space. They need to be transformed
        before being passed to a model or minimizer by `space.transform()`.

        Parameters
        ----------
        * `n_samples` [int, default=1]:
            Number of samples to be drawn from the space.

        * `random_state` [int, RandomState instance, or None (default)]:
            Set random state to something other than None for reproducible
            results.

        Returns
        -------
        * `points`: [list of lists, shape=(n_points, n_dims)]
           Points sampled from the space.
        """
        rng = check_random_state(random_state)

        # Draw
        columns = []

        for dim in self.dimensions:
            if sp_version < (0, 16):
                columns.append(dim.rvs(n_samples=n_samples))
            else:
                columns.append(dim.rvs(n_samples=n_samples, random_state=rng))

        # Transpose
        rows = []

        for i in range(n_samples):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    def transform(self, X):
        """Transform samples from the original space into a warped space.

        Note: this transformation is expected to be used to project samples
              into a suitable space for numerical optimization.

        Parameters
        ----------
        * `X` [list of lists, shape=(n_samples, n_dims)]:
            The samples to transform.

        Returns
        -------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]
            The transformed samples.
        """
        # Pack by dimension
        columns = []
        for dim in self.dimensions:
            columns.append([])

        for i in range(len(X)):
            for j in range(self.n_dims):
                columns[j].append(X[i][j])

        # Transform
        for j in range(self.n_dims):
            columns[j] = self.dimensions[j].transform(columns[j])

        # Repack as an array
        Xt = np.hstack([np.asarray(c).reshape((len(X), -1)) for c in columns])

        return Xt

    def inverse_transform(self, Xt):
        """Inverse transform samples from the warped space back to the
           original space.

        Parameters
        ----------
        * `Xt` [array of floats, shape=(n_samples, transformed_n_dims)]:
            The samples to inverse transform.

        Returns
        -------
        * `X` [list of lists, shape=(n_samples, n_dims)]
            The original samples.
        """
        # Inverse transform
        columns = []
        start = 0

        for j in range(self.n_dims):
            dim = self.dimensions[j]
            offset = dim.transformed_size

            if offset == 1:
                columns.append(dim.inverse_transform(Xt[:, start]))
            else:
                columns.append(
                    dim.inverse_transform(Xt[:, start:start+offset]))

            start += offset

        # Transpose
        rows = []

        for i in range(len(Xt)):
            r = []
            for j in range(self.n_dims):
                r.append(columns[j][i])

            rows.append(r)

        return rows

    @property
    def n_dims(self):
        """The dimensionality of the original space."""
        return len(self.dimensions)

    @property
    def transformed_n_dims(self):
        """The dimensionality of the warped space."""
        return sum([dim.transformed_size for dim in self.dimensions])

    @property
    def bounds(self):
        """The dimension bounds, in the original space."""
        b = []

        for dim in self.dimensions:
            if dim.size == 1:
                b.append(dim.bounds)
            else:
                b.extend(dim.bounds)

        return b

    def __contains__(self, point):
        """Check that `point` is within the bounds of the space."""
        for component, dim in zip(point, self.dimensions):
            if component not in dim:
                return False
        return True

    @property
    def transformed_bounds(self):
        """The dimension bounds, in the warped space."""
        b = []

        for dim in self.dimensions:
            if dim.transformed_size == 1:
                b.append(dim.transformed_bounds)
            else:
                b.extend(dim.transformed_bounds)

        return b

    @property
    def is_categorical(self):
        """Space contains exclusively categorical dimensions"""
        return all([isinstance(dim, Categorical) for dim in self.dimensions])

    def distance(self, point_a, point_b):
        """Compute distance between two points in this space.

        Parameters
        ----------
        * `a` [array]
            First point.

        * `b` [array]
            Second point.
        """
        distance = 0.
        for a, b, dim in zip(point_a, point_b, self.dimensions):
            distance += dim.distance(a, b)

        return distance

    def get_dimension(self, id):
        """Get the Dimension-object in the search-space using either
        its name or index as the lookup-id.

        Parameters
        ----------
        * `id` str or int:
            If `id` is a string then it is the name of a dimension.
            If `id` is an integer then it is the index of a dimension.

        Returns
        -------
        * (`dimension`, `index`, `name`) [tuple(Dimension, int, str)]:
            Dimension-object and its index in the search-space.
            The name is taken from the Dimension-object if not None,
            otherwise it is 'X_n' where n is the index.
        """

        if isinstance(id, str):
            # The id is assumed to be a string-name of a dimension,
            # so find its index.
            index = self.dimension_names.index(id)
        elif isinstance(id, int):
            # The id is assumed to be the index of a dimension.
            index = id
        else:
            raise ValueError("Dimension id must be either a string with the name or an integer for the index.")

        # Get the corresponding dimension-object.
        dimension = self.dimensions[index]

        # Get the name of the dimension.
        name = dimension.name

        # Use a default name for unnamed dimensions.
        # TODO: Maybe it would be better to set this in Space.__init__?
        if name is None:
            name = "X_" + str(index)

        return dimension, index, name

    def get_dimensions(self, ids=None):
        """Get the search-space dimensions with the given ids.
        The ids can be either a list of strings or integers.
        If strings then they are the names of the dimensions.
        If integers then they are the indices of the dimensions.
        If ids is None then get all the dimensions.

        Parameters
        ----------
        * `ids` [list of str or int, or None]:
            If list of `str` then it is the names of dimensions to select.
            If list of `int` then it is the indices of dimensions to select.
            If None then select all dimensions.

        Returns
        -------
        * `(dimensions, indices, names)` [tuple(list Dimension, list int, list str)]
            List of the matching dimensions from the search-space.
            List of their indices in the search-space.
            List of their names.
        """

        if ids is None:
            # We use all dimensions so the ids are just a range for all dimensions.
            ids = range(len(self.dimensions))

        # Lookup the dimensions for the ids.
        result = [self.get_dimension(id=id) for id in ids]

        # Unpack the returned lists.
        dimensions, index, names = zip(*result)

        return dimensions, index, names

    def to_dict(self, x):
        """Convert a location in the search-space from a list
        to a dict where the keys are the names of the dimensions.

        NOTE: There is a related function in `utils.point_asdict()`
        but it takes the search-space as a dict instead.
        
        Example
        -------
        If `self.dimension_names = ['height', 'width', 'color']`
        then `to_dict(x=[1, 2.0, 'red'])` returns the dict:
        `{'height': 1, 'width': 2.0, 'color': 'red'}`

        Parameters
        ----------
        * `x` [list]:
            A location in the search-space.

        Returns
        -------
        * `x_dict` [dict]
            The location `x` in the search-space wrapped in a dict.
            The keys are the names of the dimensions, and the
            values are from `x`.
        """

        x_dict = {dim_name: value for dim_name, value in zip(self.dimension_names, x)}

        return x_dict
