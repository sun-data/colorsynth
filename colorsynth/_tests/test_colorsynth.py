from typing import Callable
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


XYZ = [
    np.random.uniform(size=(3,)),
    np.random.uniform(size=(64, 64, 3)),
]


@pytest.mark.parametrize(argnames="wavelength", argvalues=wavelengths)
def test_d65_standard_illuminant(
    wavelength: u.Quantity,
):
    result = colorsynth.d65_standard_illuminant(wavelength)
    assert isinstance(result, (float, np.ndarray))


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
    axis: int,
):
    result = colorsynth.color_matching_xyz(wavelength, axis=axis)
    assert isinstance(result, (float, np.ndarray))
    assert result.shape[axis] == 3


@pytest.mark.parametrize(
    argnames="spd",
    argvalues=[
        np.random.uniform(size=(101,)),
        np.random.uniform(size=(64, 64, 101)),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        np.linspace(380, 780, num=101) * u.nm,
    ],
)
@pytest.mark.parametrize(argnames="axis", argvalues=[0, -1])
def test_XYZcie1931_from_spd(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int,
):
    result = colorsynth.XYZcie1931_from_spd(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )
    assert isinstance(result, np.ndarray)
    assert np.all(result >= 0)
    assert result.shape[axis] == 3


@pytest.mark.parametrize("XYZ", XYZ)
@pytest.mark.parametrize("axis", [-1])
def test_xyY_from_XYZ_cie(
    XYZ: np.ndarray,
    axis: int,
):
    result = colorsynth.xyY_from_XYZ_cie(XYZ, axis=axis)
    assert isinstance(result, np.ndarray)
    assert result.shape[axis] == 3


@pytest.mark.parametrize("xyY", XYZ)
@pytest.mark.parametrize("axis", [-1])
def test_XYZ_from_xyY_cie(
    xyY: np.ndarray,
    axis: int,
):
    result = colorsynth.XYZ_from_xyY_cie(xyY, axis=axis)
    assert isinstance(result, np.ndarray)
    assert result.shape[axis] == 3


@pytest.mark.parametrize("XYZ", XYZ)
@pytest.mark.parametrize("axis", [-1])
def test_XYZ_normalized(
    XYZ: np.ndarray,
    axis: int,
):
    result = colorsynth.XYZ_normalized(XYZ, axis=axis)
    assert isinstance(result, np.ndarray)
    assert result.shape[axis] == 3
    assert np.take(result, 1, axis=axis).max() <= 1


@pytest.mark.parametrize("XYZ", XYZ)
@pytest.mark.parametrize(argnames="axis", argvalues=[-1])
def test_sRGB(
    XYZ: np.ndarray,
    axis: int,
):
    result = colorsynth.sRGB(XYZ, axis=axis)
    assert isinstance(result, np.ndarray)
    assert result.shape[axis] == 3


@pytest.mark.parametrize(
    argnames="spd",
    argvalues=[
        np.random.uniform(size=(101,)),
        np.random.uniform(size=(64, 64, 101)),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        None,
        np.linspace(380, 780, num=101) * u.nm,
    ],
)
def test_rgb(
    spd: np.ndarray,
    wavelength: u.Quantity,
):
    axis = -1
    result = colorsynth.rgb(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )
    assert isinstance(result, np.ndarray)
    assert result.shape[axis] == 3


@pytest.mark.parametrize(
    argnames="spd",
    argvalues=[
        np.random.uniform(size=(101,)),
        np.random.uniform(size=(64, 64, 101)),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        None,
        np.linspace(380, 780, num=101) * u.nm,
    ],
)
def test_colorbar(
    spd: np.ndarray,
    wavelength: u.Quantity,
):
    axis = -1
    result = colorsynth.colorbar(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )
    assert isinstance(result, tuple)
    assert len(result) == 3
    for arr in result:
        assert isinstance(arr, np.ndarray)


@pytest.mark.parametrize(
    argnames="spd",
    argvalues=[
        np.random.uniform(size=(101,)),
        np.random.uniform(size=(64, 64, 101)),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        np.linspace(380, 780, num=101) * u.nm,
    ],
)
@pytest.mark.parametrize(
    argnames="spd_norm",
    argvalues=[None, np.sqrt],
)
def test_rgb_and_colorbar(
    spd: np.ndarray,
    wavelength: u.Quantity,
    spd_norm: None | Callable,
):
    axis = -1
    result = colorsynth.rgb_and_colorbar(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        spd_norm=spd_norm,
    )
    assert isinstance(result, tuple)
