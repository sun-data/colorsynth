import numpy as np
import astropy.units as u

__all__ = [
    "color_matching_x",
    "color_matching_y",
    "color_matching_z",
    "color_matching_xyz",
]


def _piecewise_gaussian(
    x: u.Quantity,
    mean: u.Quantity,
    stddev_1: u.Quantity,
    stddev_2: u.Quantity,
):
    where = x < mean
    not_where = ~where
    result = np.empty(x.shape)
    result[where] = np.exp(-np.square((x[where] - mean) / stddev_1) / 2)
    result[not_where] = np.exp(-np.square((x[not_where] - mean) / stddev_2) / 2)
    return result


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
