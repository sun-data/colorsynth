import pathlib
import numpy as np
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
    "sRGB",
]


wavelength_visible_min = 380 * u.nm
wavelength_visible_max = 700 * u.nm


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

    path = pathlib.Path(__file__).parent / "data/std65.txt"
    wavl, spd = np.genfromtxt(path, skip_header=1, unpack=True)
    wavl = wavl << u.nm

    ybar = color_matching_y(wavl)
    Y = np.trapz(x=wavl, y=ybar * spd)

    spd = spd / Y

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
    """
    spd, wavelength = np.broadcast_arrays(
        spd,
        wavelength,
        subok=True,
    )

    axis = ~(~axis % spd.ndim)

    xyz = color_matching_xyz(wavelength, axis=0)
    integrand = spd * xyz

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
