"""
Create false-color images from spectral cubes.

:mod:`colorsynth` collapses the spectral axis of a :class:`numpy.ndarray`
into red, green, and blue channels that can be displayed on a computer
monitor.
The spectrum at every point is mapped into the human visible range,
weighted by the CIE 1931 color-matching functions, integrated along the
spectral axis, and converted to the sRGB color space, so that the shape of
the spectrum controls the hue of each pixel and the total intensity
controls its brightness.

The main entry points are :func:`colorsynth.rgb`, which converts a spectral
cube into an RGB image that can be displayed with
:func:`matplotlib.pyplot.imshow` or :func:`matplotlib.pyplot.pcolormesh`,
and :func:`colorsynth.rgb_and_colorbar`, which additionally computes a 2D
colorbar relating color to wavelength and intensity.
"""

from ._colorsynth import *
