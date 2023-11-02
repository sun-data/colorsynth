import numpy as np
import astropy.units as u

__all__ = [
    "color_matching_x",
    "color_matching_y",
    "color_matching_z",
    "color_matching_xyz",
    "cie_1931_tristimulus",
    "srgb",
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


def cie_1931_tristimulus(
    spectral_radiance: np.ndarray,
    wavelength: u.Quantity,
    axis: int = -1,
) -> np.ndarray:
    """
    Calculate the CIE 1931 tristimulus values, :math:`X`, :math:`Y`, and :math:`Z`,
    for the given spectral radiance.

    Parameters
    ----------
    spectral_radiance
        the spectral radiance of an emitting source as a function of wavelength
    wavelength
        the wavelength grid corresponding to the spectral radiance.
    axis
        the wavelength axis, or the axis along which to integrate
    """
    spectral_radiance, wavelength = np.broadcast_arrays(
        spectral_radiance,
        wavelength,
        subok=True,
    )

    axis = ~(~axis % spectral_radiance.ndim)

    xyz = color_matching_xyz(wavelength, axis=0)
    integrand = spectral_radiance * xyz

    result = np.trapz(
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


def srgb(
    tristimulus: np.ndarray,
    axis: int = -1,
) -> np.ndarray:
    """
    Convert CIE 1931 tristimulus values, calculated using :func:`cie_1931_tristimulus`,
    into the `sRGB color space <https://en.wikipedia.org/wiki/SRGB>`_, the standard
    color space used on computer monitors.

    Parameters
    ----------
    tristimulus
        the CIE 1931 tristimulus values, :math:`X`, :math:`Y`, and :math:`Z`.
    axis
        the axis along which the different tristiumulus values are located

    Examples
    --------

    Plot a 2d set of random spectral radiance curves as a color image

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import colorsynth

        # Define the number of wavelength bins in our spectrum
        num = 11

        # Define an evenly-spaced grid of wavelengths
        wavelength = np.linspace(380, 780, num=num) * u.nm

        # Define a random spectral radiance cube by sampling from a uniform distribution
        spectral_radiance = np.random.uniform(size=(16, 16, num))

        # Calculate the CIE 1931 tristimulus values from the specdtral radiance
        xyz = colorsynth.cie_1931_tristimulus(spectral_radiance, wavelength)

        # Normalize the tristimulus values based on the max value of the Y parameter
        xyz = xyz / xyz[..., 1].max()

        # Convert the tristimulus values into sRGB, the standard used in most
        # computer monitors
        rgb = colorsynth.srgb(xyz)

        # Plot the result as an image
        plt.figure();
        plt.imshow(rgb);
    """
    x, y, z = np.moveaxis(tristimulus, axis, 0)

    r = +3.2406 * x - 1.5372 * y - 0.4986 * z
    g = -0.9689 * x + 1.8758 * y + 0.0415 * z
    b = +0.0557 * x - 0.2040 * y + 1.0570 * z

    result = np.stack([r, g, b], axis=axis)

    where = result <= 0.0031308
    not_where = ~where

    result[where] = 12.92 * result[where]
    result[not_where] = 1.055 * result[not_where] ** (1 / 2.4) - 0.055

    return result
