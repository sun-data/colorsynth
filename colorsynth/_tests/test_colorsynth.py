import pytest
import numpy as np
import astropy.units as u
import colorsynth


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        533 * u.nm,
        np.linspace(
            start=380 * u.nm,
            stop=780 * u.nm,
        ),
    ],
)
def test_color_matching_x(
    wavelength: u.Quantity,
):
    result = colorsynth.color_matching_x(wavelength)
    assert isinstance(result, (float, np.ndarray))
