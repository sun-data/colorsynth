import pytest
import numpy as np
import astropy.units as u
import colorsynth

wavelengths = [
    533 * u.nm,
    np.linspace(
        start=380 * u.nm,
        stop=780 * u.nm,
    ),
]


@pytest.mark.parametrize(argnames="wavelength", argvalues=wavelengths)
def test_color_matching_x(
    wavelength: u.Quantity,
):
    result = colorsynth.color_matching_x(wavelength)
    assert isinstance(result, (float, np.ndarray))


@pytest.mark.parametrize(argnames="wavelength", argvalues=wavelengths)
def test_color_matching_y(
    wavelength: u.Quantity,
):
    result = colorsynth.color_matching_y(wavelength)
    assert isinstance(result, (float, np.ndarray))


@pytest.mark.parametrize(argnames="wavelength", argvalues=wavelengths)
def test_color_matching_z(
    wavelength: u.Quantity,
):
    result = colorsynth.color_matching_z(wavelength)
    assert isinstance(result, (float, np.ndarray))


@pytest.mark.parametrize(argnames="wavelength", argvalues=wavelengths)
@pytest.mark.parametrize(argnames="axis", argvalues=[0, -1])
def test_color_matching_xyz(
    wavelength: u.Quantity,
    axis: int
):
    result = colorsynth.color_matching_xyz(wavelength, axis=axis)
    assert isinstance(result, (float, np.ndarray))
    assert result.shape[axis] == 3
