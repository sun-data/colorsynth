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

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import astropy.wcs
    import astropy.io
    import colorsynth

    hdu_list = astropy.io.fits.open(
        "https://www.lmsal.com/solarsoft/irisa/data/level2_compressed/2014/03/17/20140317Mosaic/IRISMosaic_20140317_Si1393.fits.gz"
    )
    hdu = hdu_list[0]

    wcs = astropy.wcs.WCS(hdu)
    axes = list(reversed(wcs.axis_type_names))
    axis_x = axes.index("Solar X")
    axis_y = axes.index("Solar Y")
    axis_wavelength =  axes.index("Wavelength")
    axis_xy = (axis_x, axis_y)

    spd = hdu.data

    wavelength_center = wcs.wcs.crval[~axis_wavelength] * u.AA
    hx, hy, wavelength = wcs.array_index_to_world_values(*np.indices(spd.shape))
    hx = hx * u.arcsec
    hy = hy * u.arcsec
    wavelength = wavelength * u.AA
    velocity = (wavelength - wavelength_center) * astropy.constants.c / wavelength_center
    velocity = velocity.to(u.km / u.s)

    rgb, colorbar = colorsynth.rgb_and_colorbar(
        spd=spd,
        wavelength=velocity,
        axis=axis_wavelength,
        spd_min=0,
        spd_max=1.1*np.percentile(spd, 99.5, axis=axis_xy, keepdims=True),
        wavelength_norm=lambda x: np.arcsinh(x / (25 * u.km / u.s))
    )

    with astropy.visualization.quantity_support():
        fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=[.95,.05]), constrained_layout=True)
        axs[0].pcolormesh(
            hx.mean(axis_wavelength),
            hy.mean(axis_wavelength),
            np.clip(np.moveaxis(rgb, 0, ~0), 0, 1),
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
