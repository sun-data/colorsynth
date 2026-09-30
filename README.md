# colorsynth

[![tests](https://github.com/sun-data/colorsynth/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/colorsynth/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/colorsynth/graph/badge.svg?token=8QettIppCi)](https://codecov.io/gh/sun-data/colorsynth)
[![Black](https://github.com/sun-data/colorsynth/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/colorsynth/actions/workflows/black.yml)
[![Documentation Status](https://readthedocs.org/projects/colorsynth/badge/?version=latest)](https://colorsynth.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/colorsynth.svg)](https://badge.fury.io/py/colorsynth)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23066030.svg)](https://doi.org/10.5281/zenodo.23066030)

A Python library for creating false-color images from spectral cubes:
3D Numpy arrays with two spatial axes and one spectral axis.

`colorsynth` collapses the spectral axis into red, green, and blue channels
by mapping the spectrum into the human visible range and weighting it by the
CIE 1931 color-matching functions.
The shape of the spectrum controls the hue of each pixel, and the total
intensity controls its brightness, so features like Doppler shifts are
visible directly in the image.

## Installation

`colorsynth` is available on the Python Package Index and can be installed using `pip`:
```
pip install colorsynth
```

## Quickstart

Define a spectral cube containing a Gaussian emission line at every point,
with a center wavelength that increases from left to right and a total
intensity that increases from top to bottom, and collapse it into an RGB
image using
[`colorsynth.rgb()`](https://colorsynth.readthedocs.io/en/latest/_autosummary/colorsynth.rgb.html#colorsynth.rgb):

```python
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import colorsynth

# Define a wavelength grid for the spectral axis
wavelength = np.linspace(400, 700, num=61) * u.nm

# Define a spatial grid
x = np.linspace(0, 1, num=101)[:, np.newaxis, np.newaxis]
y = np.linspace(0, 1, num=101)[np.newaxis, :, np.newaxis]

# Define a spectral cube containing a Gaussian emission line at every point
center = 420 * u.nm + (680 - 420) * u.nm * y
width = 15 * u.nm
spd = x * np.exp(-np.square((wavelength - center) / width))

# Collapse the wavelength axis of the cube into RGB channels
rgb = colorsynth.rgb(spd, wavelength, axis=~0, spd_min=0, spd_max=1)

# Display the result as a false-color image
plt.imshow(rgb)
plt.show()
```

![A synthetic spectral cube colorized with colorsynth.rgb()](https://colorsynth.readthedocs.io/en/latest/_images/index_0_0.png)

The center wavelength of the emission line appears as hue, and its total
intensity appears as brightness.

## Gallery

An Si IV 1403 A spectroheliogram captured by the [Interface Region Imaging Spectrograph](https://iris.lmsal.com) and colorized using 
[`colorsynth.rgb()`](https://colorsynth.readthedocs.io/en/latest/_autosummary/colorsynth.rgb.html#colorsynth.rgb). 
The code to create this image can be found in the [documentation](https://colorsynth.readthedocs.io/).

![IRIS spectroheliogram](https://colorsynth.readthedocs.io/en/latest/_images/index_2_1.png)

## Citation

If you use colorsynth in your research, please cite it.
The citation metadata is kept in [`CITATION.cff`](https://github.com/sun-data/colorsynth/blob/main/CITATION.cff),
which the "Cite this repository" button on GitHub can export as BibTeX or APA.

Every release of colorsynth is archived on Zenodo with its own DOI.
The concept DOI, [10.5281/zenodo.23066030](https://doi.org/10.5281/zenodo.23066030),
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of colorsynth that you used,
which is given by `importlib.metadata.version("colorsynth")`.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace `doi` with the DOI of that version.

```bibtex
@software{colorsynth,
  author = {Smart, Roy T.},
  title = {colorsynth},
  version = {X.Y.Z},
  doi = {10.5281/zenodo.23066030},
  url = {https://github.com/sun-data/colorsynth},
}
```
