Introduction
============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    colorsynth

Examples
========

The `Interface Region Imaging Spectrograph <iris.lmsal.com>`_ (IRIS), is a NASA
Small Explorer satellite that has been observing the Sun in ultraviolet since 2013.

IRIS is a scanning slit spectrograph which allows it to capture a 3D data product
in :math:`x`, :math:`y` and wavelength.
Visualizing this 3D data on a 2D computer monitor presents obvious difficulties.
With :mod:`colorsynth`, we can plot this type of data using color as a third dimension.

.. jupyter-execute::

    import shutil
    import urllib
    import pathlib
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import astropy.wcs
    import astropy.io
    import colorsynth

    archive, _ = path, headers = urllib.request.urlretrieve(
        url=r"https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2021/09/23/20210923_061339_3620108077/iris_l2_20210923_061339_3620108077_raster.tar.gz",
        filename="raster.tar.gz",
    )

    directory = pathlib.Path("raster")
    shutil.unpack_archive(filename=archive, extract_dir=directory)
    fits = list(directory.glob("*.fits"))

    hdu_list = astropy.io.fits.open(fits[0])
    hdu = hdu_list[4]

    wcs = astropy.wcs.WCS(hdu)
    wcs

    wcs = astropy.wcs.WCS(hdu)
    axes = list(reversed(wcs.axis_type_names))
    axis_x = axes.index("HPLN")
    axis_y = axes.index("HPLT")
    axis_wavelength =  axes.index("WAVE")
    axis_xy = (axis_x, axis_y)

    spd = hdu.data

    wavelength_center = hdu_list[0].header["TWAVE4"] * u.AA
    wavelength, hy, hx = wcs.array_index_to_world_values(*np.indices(spd.shape))
    hx = hx * u.deg << u.arcsec
    hy = hy * u.deg << u.arcsec
    wavelength = wavelength * u.m << u.AA
    velocity = (wavelength - wavelength_center) * astropy.constants.c / wavelength_center
    velocity = velocity.to(u.km / u.s)

    rgb, colorbar = colorsynth.rgb_and_colorbar(
        spd=spd,
        wavelength=velocity,
        axis=axis_wavelength,
        spd_min=0,
        spd_max=1.1*np.percentile(spd, 99, axis=axis_xy, keepdims=True),
    #     spd_norm=lambda x: np.nan_to_num(np.sqrt(x)),
        wavelength_min=-100 * u.km / u.s,
        wavelength_max=+100 * u.km / u.s,
        wavelength_norm=lambda x: np.arcsinh(x / (25 * u.km / u.s))
    )

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(
            ncols=2,
            figsize=(8, 8),
            gridspec_kw=dict(width_ratios=[.9,.1]),
            constrained_layout=True,
        )
        axs[0].pcolormesh(
            hx.mean(axis_wavelength),
            hy.mean(axis_wavelength),
            np.clip(np.moveaxis(rgb, axis_wavelength, ~0), 0, 1),
        )
        axs[0].set_aspect("equal")
        axs[1].pcolormesh(*colorbar)
        axs[1].yaxis.tick_right()
        axs[1].yaxis.set_label_position("right")

Bibliography
============

.. bibliography::



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
