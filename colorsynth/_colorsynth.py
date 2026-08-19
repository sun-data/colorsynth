from typing import Callable
import pathlib
import functools
import math
import numpy as np
import numba
import astropy.units as u

__all__ = [
    "wavelength_visible_min",
    "wavelength_visible_max",
    "d65_standard_illuminant",
    "color_matching_x",
    "color_matching_y",
    "color_matching_z",
    "color_matching_xyz",
    "XYZcie1931_from_spd",
    "xyY_from_XYZ_cie",
    "XYZ_from_xyY_cie",
    "XYZ_normalized",
    "sRGB",
    "rgb",
    "colorbar",
    "rgb_and_colorbar",
]


wavelength_visible_min = 380 * u.nm
wavelength_visible_max = 700 * u.nm


@functools.cache
def _d65_standard_illuminant_tabulated() -> tuple[u.Quantity, u.Quantity]:
    """
    Load the tabulated spectral power distribution of the CIE standard
    illuminant D65 and normalize it to unit luminance.

    The result is cached so that the data file is read and integrated
    only once per session.
    """
    path = pathlib.Path(__file__).parent / "data/std65.txt"
    wavl, spd = np.genfromtxt(path, skip_header=1, unpack=True)
    wavl = wavl << u.nm

    ybar = color_matching_y(wavl)
    Y = np.trapezoid(x=wavl, y=ybar * spd)

    spd = spd / Y

    return wavl, spd


def d65_standard_illuminant(
    wavelength: u.Quantity,
) -> u.Quantity:
    """
    Spectral power distribution (SPD) of the
    `CIE standard illuminant D65 <https://en.wikipedia.org/wiki/Illuminant_D65>`_,
    which corresponds to average midday light in Western/Northern Europe.

    This function interpolates the
    `tabulated SPD <https://web.archive.org/web/20171122140854/http://www.cie.co.at/publ/abst/datatables15_2004/std65.txt>`_
    provided by CIE.

    Parameters
    ----------
    wavelength
        the wavelengths at which to evaluate the spectral power distribution.

    Examples
    --------
    Plot the D65 standard illuminant over the human visible color range.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        wavelength = np.linspace(300, 780, num=1001) * u.nm
        d65 = colorsynth.d65_standard_illuminant(wavelength)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(wavelength, d65)
    """

    wavl, spd = _d65_standard_illuminant_tabulated()

    result = np.interp(
        x=wavelength,
        xp=wavl,
        fp=spd,
        left=0,
        right=0,
    )
    return result


def _piecewise_gaussian(
    x: u.Quantity,
    mean: u.Quantity,
    stddev_1: u.Quantity,
    stddev_2: u.Quantity,
) -> np.ndarray:

    unit = x.unit
    x = x.value
    mean = mean.to_value(unit)
    stddev_1 = stddev_1.to_value(unit)
    stddev_2 = stddev_2.to_value(unit)

    result = _piecewise_guassian_ufunc(x, mean, stddev_1, stddev_2)

    return result


@numba.vectorize(
    [numba.float64(numba.float64, numba.float64, numba.float64, numba.float64)],
    target="parallel",
)
def _piecewise_guassian_ufunc(
    x: float,
    mean: float,
    stddev_1: float,
    stddev_2: float,
) -> float:  # pragma: nocover

    if x < mean:
        stddev = stddev_1
    else:
        stddev = stddev_2

    a = (x - mean) / stddev

    return math.exp(-a * a / 2)


def color_matching_x(wavelength: u.Quantity) -> u.Quantity:
    r"""
    The CIE 1931 :math:`\overline{x}(\lambda)` color matching function.

    Calculated using the piecewise Gaussian fit method described in
    :cite:t:`Wyman2013`

    Parameters
    ----------
    wavelength
        the wavelengths at which to evaluate the color-matching function

    Examples
    --------

    Plot :math:`\overline{x}(\lambda)` over the entire human visible wavelength range

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        wavelength = np.linspace(380, 780, num=101) * u.nm
        xbar = colorsynth.color_matching_x(wavelength)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(wavelength, xbar)
    """
    g = _piecewise_gaussian
    term_1 = 1.056 * g(
        x=wavelength,
        mean=599.8 * u.nm,
        stddev_1=37.9 * u.nm,
        stddev_2=31.0 * u.nm,
    )
    term_2 = 0.362 * g(
        x=wavelength,
        mean=442.0 * u.nm,
        stddev_1=16.0 * u.nm,
        stddev_2=26.7 * u.nm,
    )
    term_3 = -0.065 * g(
        x=wavelength,
        mean=501.1 * u.nm,
        stddev_1=20.4 * u.nm,
        stddev_2=26.2 * u.nm,
    )
    result = term_1 + term_2 + term_3
    return result


def color_matching_y(wavelength: u.Quantity) -> u.Quantity:
    r"""
    The CIE 1931 :math:`\overline{y}(\lambda)` color matching function.

    Calculated using the piecewise Gaussian fit method described in
    :cite:t:`Wyman2013`

    Parameters
    ----------
    wavelength
        the wavelengths at which to evaluate the color-matching function

    Examples
    --------

    Plot :math:`\overline{y}(\lambda)` over the entire human visible wavelength range

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        wavelength = np.linspace(380, 780, num=101) * u.nm
        ybar = colorsynth.color_matching_y(wavelength)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(wavelength, ybar)
    """
    g = _piecewise_gaussian
    term_1 = 0.821 * g(
        x=wavelength,
        mean=568.8 * u.nm,
        stddev_1=46.9 * u.nm,
        stddev_2=40.5 * u.nm,
    )
    term_2 = 0.286 * g(
        x=wavelength,
        mean=530.9 * u.nm,
        stddev_1=16.3 * u.nm,
        stddev_2=31.1 * u.nm,
    )
    result = term_1 + term_2
    return result


def color_matching_z(wavelength: u.Quantity) -> u.Quantity:
    r"""
    The CIE 1931 :math:`\overline{z}(\lambda)` color matching function.

    Calculated using the piecewise Gaussian fit method described in
    :cite:t:`Wyman2013`

    Parameters
    ----------
    wavelength
        the wavelengths at which to evaluate the color-matching function

    Examples
    --------

    Plot :math:`\overline{z}(\lambda)` over the entire human visible wavelength range

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        wavelength = np.linspace(380, 780, num=101) * u.nm
        zbar = colorsynth.color_matching_z(wavelength)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(wavelength, zbar)
    """
    g = _piecewise_gaussian
    term_1 = 1.217 * g(
        x=wavelength,
        mean=437.0 * u.nm,
        stddev_1=11.8 * u.nm,
        stddev_2=36.0 * u.nm,
    )
    term_2 = 0.681 * g(
        x=wavelength,
        mean=459.0 * u.nm,
        stddev_1=26.0 * u.nm,
        stddev_2=13.8 * u.nm,
    )
    result = term_1 + term_2
    return result


def color_matching_xyz(
    wavelength: u.Quantity,
    axis: int = -1,
) -> u.Quantity:
    r"""
    The CIE 1931 :math:`\overline{x}(\lambda)`, :math:`\overline{y}(\lambda)`,
    and :math:`\overline{z}(\lambda)` color matching functions.

    Stack the results of :func:`color_matching_x`, :func:`color_matching_y`,
    and :func:`color_matching_z` into a single array.

    Parameters
    ----------
    wavelength
        the wavelengths at which to evaluate the color-matching function
    axis
        the axis in the result along which the :math:`\overline{x}(\lambda)`,
        :math:`\overline{y}(\lambda)`, and :math:`\overline{z}(\lambda)` arrays
        are stacked.

    Examples
    --------

    Plot :math:`\overline{x}(\lambda)`, :math:`\overline{y}(\lambda)`,
    and :math:`\overline{z}(\lambda)` over the entire human visible wavelength range.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        wavelength = np.linspace(380, 780, num=101) * u.nm
        xyz = colorsynth.color_matching_xyz(wavelength, axis=0)

        with astropy.visualization.quantity_support():
            plt.figure()
            plt.plot(wavelength, xyz[0], color="red", label=r"$\overline{x}(\lambda)$")
            plt.plot(wavelength, xyz[1], color="green", label=r"$\overline{y}(\lambda)$")
            plt.plot(wavelength, xyz[2], color="blue", label=r"$\overline{z}(\lambda)$")
            plt.legend()
    """
    x = color_matching_x(wavelength)
    y = color_matching_y(wavelength)
    z = color_matching_z(wavelength)
    result = np.stack([x, y, z], axis=axis)
    return result


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    r"""
    Quadrature weights :math:`w_i` such that :math:`\sum_i w_i y_i` is
    equal to :func:`numpy.trapezoid` of :math:`y` over the grid :math:`x`.

    Parameters
    ----------
    x
        the grid of the integration variable.
    """
    result = np.zeros(x.shape)
    if x.size >= 2:
        d = np.diff(x)
        result[0] = d[0] / 2
        result[-1] = d[-1] / 2
        result[1:-1] = (x[2:] - x[:-2]) / 2
    return result


def _validate_spd_wavelength(
    spd: np.ndarray,
    wavelength: np.ndarray,
    axis: int,
) -> None:
    """
    Check that `spd` and `wavelength` are broadcastable against each other
    and that `wavelength` varies along `axis` of the broadcast shape,
    raising an informative :class:`ValueError` otherwise.

    Parameters
    ----------
    spd
        the spectral power distribution of an emitting source as a function of wavelength
    wavelength
        the wavelength grid corresponding to the spectral power distribution.
    axis
        the wavelength axis, or the axis along which to integrate
    """
    spd = np.asanyarray(spd)
    wavelength = np.asanyarray(wavelength)

    ndim = max(spd.ndim, wavelength.ndim)
    if not -ndim <= axis < ndim:
        raise ValueError(
            f"{axis=} is out of bounds for arrays with {ndim} dimension(s)."
        )

    try:
        shape = np.broadcast_shapes(spd.shape, wavelength.shape)
    except ValueError as e:
        raise ValueError(
            f"spd and wavelength are not broadcastable against each other, "
            f"{spd.shape=}, {wavelength.shape=}."
        ) from e

    axis_ = axis % len(shape)
    axis_wavelength = axis_ - (len(shape) - wavelength.ndim)
    num_wavelength = wavelength.shape[axis_wavelength] if axis_wavelength >= 0 else 1

    if num_wavelength != shape[axis_]:
        raise ValueError(
            f"wavelength must vary along `axis`, but wavelength has length "
            f"{num_wavelength} along {axis=} of the broadcast shape {shape}. "
            f"Reshape wavelength so that its values lie along `axis`, "
            f"for example using numpy.newaxis."
        )


def _wavelength_varies_only_along(
    wavelength: np.ndarray,
    axis: int,
    ndim: int,
) -> bool:
    """
    Check if `wavelength` can vary only along `axis` of the broadcast shape,
    which is the case if every other dimension of `wavelength` either has
    length one or is a stride-zero broadcast view.

    Parameters
    ----------
    wavelength
        the wavelength grid to inspect.
    axis
        the integration axis, relative to the broadcast shape.
    ndim
        the number of dimensions of the broadcast shape.
    """
    offset = ndim - wavelength.ndim
    axis = axis % ndim
    for i in range(wavelength.ndim):
        if i + offset == axis:
            continue
        if wavelength.shape[i] != 1 and wavelength.strides[i] != 0:
            return False
    return True


@numba.njit(parallel=True, cache=True)
def _XYZ_from_spd_weighted(
    spd: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:  # pragma: nocover
    """
    Contract a 2D spectral power distribution with the color matching
    functions premultiplied by quadrature weights.

    Parameters
    ----------
    spd
        the spectral power distribution with shape ``(num, num_wavelength)``.
    xyz
        the weighted color matching functions with shape ``(num_wavelength, 3)``.
    """
    num, num_wavelength = spd.shape
    result = np.empty((num, 3), dtype=spd.dtype)
    for i in numba.prange(num):
        X = 0.0
        Y = 0.0
        Z = 0.0
        for j in range(num_wavelength):
            s = spd[i, j]
            X += s * xyz[j, 0]
            Y += s * xyz[j, 1]
            Z += s * xyz[j, 2]
        result[i, 0] = X
        result[i, 1] = Y
        result[i, 2] = Z
    return result


def _XYZcie1931_from_spd_1d(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    """
    Fast path of :func:`XYZcie1931_from_spd` for the case where the
    wavelength grid varies only along the integration axis.

    Evaluates the color matching functions on the 1D wavelength grid,
    folds the trapezoidal quadrature weights into them, and contracts the
    result with the spectral power distribution using a compiled kernel,
    so that no array larger than `spd` is ever allocated.

    Parameters
    ----------
    spd
        the spectral power distribution of an emitting source as a function of wavelength
    wavelength
        the wavelength grid corresponding to the spectral power distribution.
    axis
        the integration axis, normalized to be negative.
    shape
        the shape of `spd` and `wavelength` broadcast against each other.
    """
    spd = np.broadcast_to(spd, shape, subok=True)

    index: list[int | slice] = [0] * len(shape)
    index[axis] = slice(None)
    wavelength = np.broadcast_to(wavelength, shape, subok=True)[tuple(index)]

    unit = None
    if isinstance(spd, u.Quantity):
        unit = spd.unit
        spd = spd.value
    if isinstance(wavelength, u.Quantity):
        unit = wavelength.unit if unit is None else unit * wavelength.unit

    xyz = color_matching_xyz(wavelength, axis=~0)

    wavelength_value = (
        wavelength.value if isinstance(wavelength, u.Quantity) else wavelength
    )
    xyz = xyz * _trapezoid_weights(wavelength_value)[..., np.newaxis]

    if not np.issubdtype(spd.dtype, np.floating):
        spd = spd.astype(float)

    spd = np.moveaxis(spd, axis, ~0)
    result = _XYZ_from_spd_weighted(spd.reshape(-1, spd.shape[~0]), xyz)
    result = result.reshape(spd.shape[:~0] + (3,))
    result = np.moveaxis(result, ~0, axis)

    if unit is not None:
        result = result << unit
    return result


def XYZcie1931_from_spd(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int = -1,
) -> np.ndarray:
    """
    Calculate the CIE 1931 tristimulus values, :math:`XYZ`,
    for the given spectral power distribution.

    Parameters
    ----------
    spd
        the spectral power distribution of an emitting source as a function of wavelength
    wavelength
        the wavelength grid corresponding to the spectral power distribution.
        Must be sorted to yield positive :math:`XYZ` values
    axis
        the wavelength axis, or the axis along which to integrate

    Notes
    -----
    If `wavelength` varies only along `axis`, which is the ordinary case,
    this function evaluates the color matching functions on the 1D
    wavelength grid and computes the integral as a weighted contraction,
    which never allocates an array larger than `spd`.
    Otherwise, it falls back to broadcasting `wavelength` against `spd`,
    which allocates several arrays three times larger than `spd`.
    """
    spd = np.asanyarray(spd)
    wavelength = np.asanyarray(wavelength)

    _validate_spd_wavelength(spd, wavelength, axis)

    shape = np.broadcast_shapes(spd.shape, wavelength.shape)

    axis = ~(~axis % len(shape))

    if _wavelength_varies_only_along(wavelength, axis, len(shape)):
        return _XYZcie1931_from_spd_1d(spd, wavelength, axis, shape)

    spd, wavelength = np.broadcast_arrays(
        spd,
        wavelength,
        subok=True,
    )

    xyz = color_matching_xyz(wavelength, axis=0)
    integrand = spd * xyz

    result = np.trapezoid(
        x=wavelength,
        y=integrand,
        axis=axis,
    )
    result = np.moveaxis(
        a=result,
        source=0,
        destination=axis,
    )
    return result


def xyY_from_XYZ_cie(
    XYZ: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Convert from a CIE :math:`XYZ` color space to a :math:`xyY` color space

    Parameters
    ----------
    XYZ
        color values in a CIE :math:`XYZ` color space to be converted
    axis
        logical axis along which the :math:`XYZ` values are distributed
    """
    XYZ_sum = XYZ.sum(axis)
    X, Y, Z = np.moveaxis(XYZ, source=axis, destination=0)
    x = X / XYZ_sum
    y = Y / XYZ_sum
    result = np.stack([x, y, Y], axis=axis)
    return result


def XYZ_from_xyY_cie(
    xyY: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Convert from a CIE :math:`xyY` color space to a :math:`XYZ` color space

    Parameters
    ----------
    xyY
        color values in a CIE :math:`xyY` color space to be converted
    axis
        logical axis along which the :math:`xyY` values are distributed
    """
    x, y, Y = np.moveaxis(xyY, source=axis, destination=0)
    r = Y / y
    X = r * x
    Z = r * (1 - x - y)
    result = np.stack([X, Y, Z], axis=axis)
    return result


def XYZ_normalized(
    XYZ: np.ndarray,
    axis: int = -1,
    axis_max: None | int | tuple[int, ...] = None,
) -> np.ndarray:
    """
    Normalize the luminance of a vector in the CIE 1931 :math:`XYZ` color space.

    This function converts to the `xyY` color space,
    divides :math:`Y` by its maximum so that the largest luminance is one,
    and then converts back into the `XYZ` color space.

    Parameters
    ----------
    XYZ
        color values in a CIE 1931 :math:`XYZ` color space to be normalized
    axis
        the axis along which the color space values are distributed
    axis_max
        the axis or axes of `XYZ` along which to compute the maximum
        luminance, ignoring NaN values.
        If :obj:`None` (the default), the maximum is computed over the
        entire array, so the result has a single global maximum luminance
        of one.
        For a stack of images, pass the axes of each image to normalize
        every image in the stack independently.
        May not contain `axis`.
    """
    ndim = XYZ.ndim
    axis_ = axis % ndim

    if axis_max is None:
        axis_max_ = None
    else:
        if isinstance(axis_max, int):
            axis_max = (axis_max,)
        axis_max_normalized = tuple(a % ndim for a in axis_max)
        if axis_ in axis_max_normalized:
            raise ValueError(
                f"{axis_max=} may not contain the color-component axis, {axis=}."
            )
        axis_max_ = tuple(a - (a > axis_) for a in axis_max_normalized)

    xyY = xyY_from_XYZ_cie(XYZ, axis=axis)
    x, y, Y = np.moveaxis(xyY, source=axis, destination=0)
    Y = Y / np.nanmax(Y, axis=axis_max_, keepdims=True)
    xyY = np.stack([x, y, Y], axis=axis)
    return XYZ_from_xyY_cie(xyY, axis=axis)


def sRGB(
    XYZ: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Convert CIE 1931 tristimulus values, calculated using :func:`XYZcie1931_from_spd`,
    into the `sRGB color space <https://en.wikipedia.org/wiki/SRGB>`_, the standard
    color space used on computer monitors.

    Parameters
    ----------
    XYZ
        the CIE 1931 tristimulus values, :math:`XYZ`.
    axis
        the axis along which the different tristimulus values are arranged

    Examples
    --------

    Plot a 2d set of random spectral power distribution curves as a color image

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import colorsynth

        # Define the number of wavelength bins in our spectrum
        num = 11

        # Define an evenly-spaced grid of wavelengths
        wavelength = np.linspace(380, 780, num=num) * u.nm

        # Define a random spectral power distribution cube by sampling from a uniform distribution
        spd = np.random.uniform(size=(16, 16, num))

        # Calculate the CIE 1931 tristimulus values from the spectral power distribution
        XYZ = colorsynth.XYZcie1931_from_spd(spd, wavelength)

        # Normalize the tristimulus values based on the max value of the Y parameter
        XYZ = XYZ / XYZ[..., 1].max()

        # Convert the tristimulus values into sRGB, the standard used in most
        # computer monitors
        rgb = colorsynth.sRGB(XYZ)

        # Plot the result as an image
        plt.figure();
        plt.imshow(rgb);

    |

    Plot the sRGB `color gamut <https://en.wikipedia.org/wiki/Gamut>`_, the complete
    subset of colors that can be reproduced accurately with sRGB.

    .. jupyter-execute::

        # Define a grid of CIE xy values
        x = np.linspace(0, 0.7, num=1000)[:, np.newaxis]
        y = np.linspace(0, 0.7, num=1001)[np.newaxis, :]

        # Define a very small value for the luminance,
        # so that the gamut is as large as possible
        Y = 1e-3

        # Define an axis which represents the
        # components of the color vectors
        axis = -1

        # Create a CIE 1931 xyY color vector
        xyY = np.stack(np.broadcast_arrays(x, y, Y), axis=axis)

        # Convert the color space from CIE 1931 xyY to XYZ
        XYZ = colorsynth.XYZ_from_xyY_cie(xyY, axis=axis)

        # Convert the color space again from CIE 1931 XYZ
        # to our target, sRGB.
        rgb = colorsynth.sRGB(XYZ, axis=axis)

        # Find the pixels that are within the sRGB gamut
        # by checking if they are finite, and if they lie within the range 0-1.
        where_nan = ~np.all(np.isfinite(rgb), axis=axis, keepdims=True)
        where_invalid = ~np.all((0 <= rgb) & (rgb <= 1), axis=axis, keepdims=True)
        where_outside = where_nan | where_invalid
        where_outside = np.broadcast_to(where_outside, rgb.shape)
        where_inside = ~where_outside

        # Set the pixels outside the gamut to gray
        rgb[where_outside] = 0.5

        # Scale the RGB values inside the gamut to the most saturated
        # color possible
        rgb[where_inside] = (rgb / np.max(rgb, axis=axis, keepdims=True))[where_inside]

        # plot the sRGB gamut
        plt.figure();
        plt.pcolormesh(
            *np.broadcast_arrays(x, y),
            np.moveaxis(rgb, source=axis, destination=-1),
        );
        plt.xlabel("CIE 1931 $x$");
        plt.ylabel("CIE 1931 $y$");

    |

    Plot the response curves of the :math:`R`, :math:`G`, and :math:`B` to
    a constant spectral power distribution

    .. jupyter-execute::

        # Define an evenly-spaced grid of wavelengths
        wavelength = np.linspace(380, 780, num=101) * u.nm

        spd = np.diagflat(np.ones(wavelength.shape))

        # Calculate the CIE 1931 tristimulus values from the spectral power distribution
        XYZ = colorsynth.XYZcie1931_from_spd(spd, wavelength[..., np.newaxis], axis=0)

        # Normalize the tristimulus values based on the max value of the Y parameter
        XYZ = XYZ / XYZ.max(axis=1, keepdims=True)
        XYZ = XYZ * np.array([0.9505, 1.0000, 1.0890])[..., np.newaxis]

        # Convert the tristimulus values into sRGB
        r, g, b = np.clip(colorsynth.sRGB(XYZ, axis=0), 0, 10)

        plt.figure();
        plt.plot(wavelength, r, color="red");
        plt.plot(wavelength, g, color="green");
        plt.plot(wavelength, b, color="blue");


    """
    X, Y, Z = np.moveaxis(XYZ, axis, 0)

    r = +3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b = +0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    result = np.stack([r, g, b], axis=axis)

    where = result <= 0.0031308
    not_where = ~where

    result[where] = 12.92 * result[where]
    result[not_where] = 1.055 * result[not_where] ** (1 / 2.4) - 0.055

    return result


def _bounds_normalize(
    a: np.ndarray,
    axis: int,
    vmin: None | np.ndarray,
    vmax: None | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    axis_orthogonal = list(range(a.ndim))
    axis_orthogonal.pop(axis)
    axis_orthogonal = tuple(axis_orthogonal)

    if vmin is None:
        vmin = np.nanmin(a, axis=axis_orthogonal, keepdims=True)
    if vmax is None:
        vmax = np.nanmax(a, axis=axis_orthogonal, keepdims=True)

    return vmin, vmax


def _transform_normalize(
    a: np.ndarray,
    axis: int,
    vmin: None | np.ndarray,
    vmax: None | np.ndarray,
    norm: None | Callable[[np.ndarray], np.ndarray],
) -> Callable[[np.ndarray], np.ndarray]:

    vmin, vmax = _bounds_normalize(
        a=a,
        axis=axis,
        vmin=vmin,
        vmax=vmax,
    )

    if norm is None:
        norm = lambda x: x

    def result(x: np.ndarray):
        vmin_normalized = norm(vmin)
        vmax_normalized = norm(vmax)
        x_normalized = norm(x)
        x = (x_normalized - vmin_normalized) / (vmax_normalized - vmin_normalized)
        x = np.nan_to_num(x, copy=False)
        return x

    return result


def _transform_wavelength(
    wavelength: u.Quantity,
    axis: int,
    vmin: None | np.ndarray,
    vmax: None | np.ndarray,
    norm: None | Callable[[np.ndarray], np.ndarray],
):
    if vmin is None:
        vmin = np.nanmin(wavelength)
    if vmax is None:
        vmax = np.nanmax(wavelength)

    transform_normalize = _transform_normalize(
        a=wavelength,
        axis=axis,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
    )

    def result(x: u.Quantity):
        x = transform_normalize(x)
        wavelength_visible_range = wavelength_visible_max - wavelength_visible_min
        x = wavelength_visible_range * x + wavelength_visible_min
        return x

    return result


def _transform_spd_wavelength(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int,
    spd_min: None | np.ndarray,
    spd_max: None | np.ndarray,
    spd_norm: None | Callable[[np.ndarray], np.ndarray],
    wavelength_min: None | u.Quantity,
    wavelength_max: None | u.Quantity,
    wavelength_norm: None | Callable[[u.Quantity], u.Quantity],
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    transform_wavelength = _transform_wavelength(
        wavelength=wavelength,
        axis=axis,
        vmin=wavelength_min,
        vmax=wavelength_max,
        norm=wavelength_norm,
    )

    transform_spd_normalize = _transform_normalize(
        a=spd,
        axis=axis,
        vmin=spd_min,
        vmax=spd_max,
        norm=None,
    )

    def transform_spd_wavelength(x: np.ndarray, w: u.Quantity):
        w = transform_wavelength(w)
        d65 = d65_standard_illuminant(w)
        x = transform_spd_normalize(x)
        if spd_norm is not None:
            x = spd_norm(x)
        x = d65 * x
        return x, w

    return transform_spd_wavelength


def rgb(
    spd: np.ndarray,
    wavelength: None | u.Quantity = None,
    axis: int = -1,
    spd_min: None | np.ndarray = None,
    spd_max: None | np.ndarray = None,
    spd_norm: None | Callable[[np.ndarray], np.ndarray] = None,
    wavelength_min: None | u.Quantity = None,
    wavelength_max: None | u.Quantity = None,
    wavelength_norm: None | Callable[[u.Quantity], u.Quantity] = None,
):
    """
    Convert a given spectral power distribution into a RGB array that can
    be plotted with matplotlib.

    Parameters
    ----------
    spd
        a spectral power distribution to be converted into a RGB array
    wavelength
        The wavelength array corresponding to the spectral power distribution.
        If :obj:`None`, the wavelength is assumed to be evenly sampled across
        the human visible color range.
    axis
        the logical axis corresponding to changing wavelength,
        or the axis along which to integrate the spectral power distribution
    spd_min
        the value of the spectral power distribution representing minimum
        intensity.
    spd_max
        the value of the spectral power distribution representing maximum
        intensity.
    spd_norm
        an optional function to transform the spectral power distribution
        values before mapping to RGB
    wavelength_min
        the wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        the wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        an optional function to transform the wavelength values before they
        are mapped into the human visible color range.

    Examples
    --------
    Colorize a random, 3D numpy array.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import colorsynth

        # Create a uniform random 3D numpy array
        a = np.random.uniform(low=0, high=1, size=(16, 16, 11))

        # Colorize the 3D numpy array
        rgb = colorsynth.rgb(a)

        # Plot the resulting RGB image
        fig, ax = plt.subplots(constrained_layout=True)
        ax.imshow(rgb);
    """
    if wavelength is None:
        shape_wavelength = [1] * spd.ndim
        shape_wavelength[axis] = -1
        wavelength = np.linspace(0, 1, num=spd.shape[axis])
        wavelength = wavelength.reshape(shape_wavelength)

    _validate_spd_wavelength(spd, wavelength, axis)

    transform_spd_wavelength = _transform_spd_wavelength(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )

    spd, wavelength = transform_spd_wavelength(spd, wavelength)

    XYZ = XYZcie1931_from_spd(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )

    RGB = sRGB(
        XYZ=XYZ,
        axis=axis,
    )

    RGB = RGB.to_value(u.dimensionless_unscaled)

    max_rgb = RGB.max(axis, keepdims=True)

    max_rgb = np.maximum(max_rgb, 1)

    RGB = RGB / max_rgb

    RGB = np.clip(RGB, 0, None)

    return RGB


def colorbar(
    spd: np.ndarray,
    wavelength: None | u.Quantity = None,
    axis: int = -1,
    axis_intensity: int = 0,
    axis_wavelength: int = 1,
    spd_min: None | np.ndarray = None,
    spd_max: None | np.ndarray = None,
    spd_norm: None | Callable[[np.ndarray], np.ndarray] = None,
    wavelength_min: None | u.Quantity = None,
    wavelength_max: None | u.Quantity = None,
    wavelength_norm: None | Callable[[u.Quantity], u.Quantity] = None,
    num_intensity: int = 101,
    squeeze: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the colorbar corresponding to calling :func:`rgb` with these
    same arguments.

    The return value from this function is designed to be used directly
    by :func:`matplotlib.pyplot.pcolormesh`.

    Parameters
    ----------
    spd
        a spectral power distribution to be converted into a RGB array
    wavelength
        The wavelength array corresponding to the spectral power distribution.
        If :obj:`None`, the wavelength is assumed to be evenly sampled across
        the human visible color range.
    axis
        The logical axis corresponding to changing wavelength,
        or the axis along which to integrate the spectral power distribution
    axis_intensity
        The index of  new logical axis in the result which corresponds to changing
        spectral radiance.
    axis_wavelength
        The index of a new logical axis in the result which corresponds to
        changing wavelength.
    spd_min
        the value of the spectral power distribution representing minimum
        intensity.
    spd_max
        the value of the spectral power distribution representing maximum
        intensity.
    spd_norm
        an optional function to transform the spectral power distribution
        values before mapping to RGB
    wavelength_min
        the wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        the wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        an optional function to transform the wavelength values before they
        are mapped into the human visible color range.
    num_intensity
        the number of intensity samples in the colorbar.
    squeeze
        A boolean flag indicating whether to remove singleton dimensions
        from the result.
        If you're just making a single colorbar, this should be :obj:`True`
        (the default) so :func:`matplotlib.pyplot.pcolormesh` will work correctly.
        If you're making a stack of colorbars, you might want to set this to
        :obj:`False` so that you don't lose track of axis meanings.

    Examples
    --------

    Plot the colorbar corresponding to a random, 3D cube.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import colorsynth

        # Define a random 3d cube
        a = np.random.uniform(
            low=0,
            high=1000,
            size=(16, 16, 11),
        ) * u.photon

        # Define wavelength axis
        wavelength = np.linspace(
            start=100 * u.AA,
            stop=200 * u.AA,
            num=a.shape[~0],
        )

        # Compute the colorbar corresponding to the random 3d cube.
        colorbar = colorsynth.colorbar(
            spd=a,
            wavelength=wavelength,
            axis=~0,
        )

        # Plot the colorbar
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            plt.pcolormesh(*colorbar)
    """
    if wavelength is None:
        shape_wavelength = [1] * spd.ndim
        shape_wavelength[axis] = -1
        wavelength = np.linspace(0, 1, num=spd.shape[axis])
        wavelength = wavelength.reshape(shape_wavelength)

    _validate_spd_wavelength(spd, wavelength, axis)

    shape = np.broadcast_shapes(spd.shape, wavelength.shape)
    ndim = len(shape)
    axis_ = ~range(ndim)[~axis]

    shape_singleton = (1,) * ndim

    transform_spd_wavelength = _transform_spd_wavelength(
        spd=spd,
        wavelength=wavelength,
        axis=axis_,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )

    spd_min_, spd_max_ = _bounds_normalize(
        a=spd,
        axis=axis,
        vmin=spd_min,
        vmax=spd_max,
    )

    spd_min_ = np.broadcast_to(
        array=spd_min_,
        shape=np.broadcast_shapes(np.shape(spd_min_), shape_singleton),
        subok=True,
    )
    spd_max_ = np.broadcast_to(
        array=spd_max_,
        shape=np.broadcast_shapes(np.shape(spd_max_), shape_singleton),
        subok=True,
    )

    spd_min_ = np.nanmin(spd_min_, axis=axis, keepdims=True)
    spd_max_ = np.nanmax(spd_max_, axis=axis, keepdims=True)

    intensity = np.linspace(
        start=0,
        stop=spd_max_ - spd_min_,
        num=num_intensity,
    )

    intensity = intensity[np.newaxis, :]

    wavelength2 = wavelength[np.newaxis, np.newaxis]
    wavelength2 = np.swapaxes(wavelength2, 0, axis_)

    shape_cbar = np.broadcast_shapes(
        intensity.shape,
        wavelength.shape,
        wavelength2.shape,
    )

    shape_index = [1] * max(wavelength.ndim, -axis_)
    shape_index[axis_] = shape[axis_]
    index = np.arange(shape[axis_]).reshape(shape_index)
    index2 = np.swapaxes(index[np.newaxis, np.newaxis], 0, axis_)

    cbar = np.zeros(shape_cbar)
    cbar[np.broadcast_to(index == index2, shape_cbar)] = 1
    cbar = cbar * intensity + spd_min_

    spd_, wavelength_ = transform_spd_wavelength(cbar, wavelength)

    XYZ = XYZcie1931_from_spd(spd_, wavelength_, axis=axis_)
    RGB = sRGB(XYZ, axis=axis_)

    RGB = RGB.to_value(u.dimensionless_unscaled)

    RGB = np.clip(RGB, 0, 1)

    wavelength2, intensity = np.broadcast_arrays(wavelength2, intensity, subok=True)

    if squeeze:
        intensity = intensity.squeeze()
        wavelength2 = wavelength2.squeeze()
        RGB = RGB.squeeze()

    source = (0, 1)
    destination = (axis_wavelength, axis_intensity)
    wavelength2 = np.moveaxis(wavelength2, source, destination)
    intensity = np.moveaxis(intensity, source, destination)
    RGB = np.moveaxis(RGB, source, destination)

    return intensity, wavelength2, RGB


def rgb_and_colorbar(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int = -1,
    spd_min: None | np.ndarray = None,
    spd_max: None | np.ndarray = None,
    spd_norm: None | Callable[[np.ndarray], np.ndarray] = None,
    wavelength_min: None | u.Quantity = None,
    wavelength_max: None | u.Quantity = None,
    wavelength_norm: None | Callable[[u.Quantity], u.Quantity] = None,
    **kwargs_colorbar,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convenience function that calls :func:`rgb` and :func:`colorbar` and
    returns the results as a tuple.

    Parameters
    ----------
    spd
        a spectral power distribution to be converted into a RGB array
    wavelength
        the wavelength array corresponding to the spectral power distribution
    axis
        the logical axis corresponding to changing wavelength,
        or the axis along which to integrate the spectral power distribution
    spd_min
        the value of the spectral power distribution representing minimum
        intensity.
    spd_max
        the value of the spectral power distribution representing maximum
        intensity.
    spd_norm
        an optional function to transform the spectral power distribution
        values before mapping to RGB
    wavelength_min
        the wavelength value that is mapped to the minimum wavelength of the
        human visible color range, 380 nm.
    wavelength_max
        the wavelength value that is mapped to the maximum wavelength of the
        human visible color range, 700 nm
    wavelength_norm
        an optional function to transform the wavelength values before they
        are mapped into the human visible color range.
    kwargs_colorbar
        Any additional keyword arguments needed by :func:`colorbar`.
    """

    kwargs = dict(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_min=spd_min,
        spd_max=spd_max,
        spd_norm=spd_norm,
        wavelength_min=wavelength_min,
        wavelength_max=wavelength_max,
        wavelength_norm=wavelength_norm,
    )

    RGB = rgb(**kwargs)
    cbar = colorbar(**kwargs, **kwargs_colorbar)

    return RGB, cbar
