from typing import Callable
import pytest
import numpy as np
import astropy.units as u
import colorsynth

rng = np.random.default_rng(42)

wavelengths = [
    533 * u.nm,
    np.linspace(
        start=380 * u.nm,
        stop=780 * u.nm,
    ),
]


XYZ = [
    rng.uniform(size=(3,)),
    rng.uniform(size=(64, 64, 3)),
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


def test_color_matching_y_peak():
    result = colorsynth.color_matching_y(555 * u.nm)
    assert np.allclose(result, 1)


def test_color_matching_xyz_outside_tabulated_range():
    wavelength = [200, 900] * u.nm
    result = colorsynth.color_matching_xyz(wavelength)
    assert np.all(result == 0)


def test_d65_white_point():
    wavelength = np.linspace(360, 830, num=2001) * u.nm
    spd = colorsynth.d65_standard_illuminant(wavelength)
    XYZ = colorsynth.XYZcie1931_from_spd(spd, wavelength)
    xyY = colorsynth.xyY_from_XYZ_cie(XYZ)
    assert np.allclose(xyY[:2], [0.31272, 0.32903], atol=1e-3)


@pytest.mark.parametrize(
    argnames="spd,wavelength,axis",
    argvalues=[
        (
            rng.uniform(size=(101,)),
            np.linspace(380, 780, num=101) * u.nm,
            0,
        ),
        (
            rng.uniform(size=(101,)),
            np.linspace(380, 780, num=101) * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(64, 64, 101)),
            np.linspace(380, 780, num=101) * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(101, 64, 64)),
            np.linspace(380, 780, num=101)[:, np.newaxis, np.newaxis] * u.nm,
            0,
        ),
    ],
)
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


@pytest.mark.parametrize(
    argnames="spd,wavelength,axis",
    argvalues=[
        (
            rng.uniform(size=(16, 17, 101)),
            np.linspace(380, 780, num=101) * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(101, 16, 17)),
            np.linspace(380, 780, num=101)[:, np.newaxis, np.newaxis] * u.nm,
            0,
        ),
        (
            rng.uniform(size=(16, 17, 101)) * u.photon,
            np.linspace(380, 780, num=101) * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(16, 17, 101)),
            np.geomspace(380, 780, num=101) * u.nm,
            -1,
        ),
        (
            *np.broadcast_arrays(
                rng.uniform(size=(16, 17, 101)),
                np.linspace(380, 780, num=101) * u.nm,
                subok=True,
            ),
            -1,
        ),
        (
            rng.uniform(size=(16, 17, 101)),
            np.linspace(380, 780, num=101) * u.nm
            + np.linspace(0, 10, num=16)[:, np.newaxis, np.newaxis] * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(16, 17, 1)),
            np.array([533]) * u.nm,
            -1,
        ),
        (
            rng.integers(0, 100, size=(16, 17, 101)),
            np.linspace(380, 780, num=101) * u.nm,
            -1,
        ),
    ],
)
def test_XYZcie1931_from_spd_trapezoid_equivalence(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int,
):
    result = colorsynth.XYZcie1931_from_spd(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )

    spd_, wavelength_ = np.broadcast_arrays(spd, wavelength, subok=True)
    axis_ = ~(~axis % spd_.ndim)
    xyz = colorsynth.color_matching_xyz(wavelength_, axis=0)
    expected = np.trapezoid(x=wavelength_, y=spd_ * xyz, axis=axis_)
    expected = np.moveaxis(expected, 0, axis_)

    assert result.shape == expected.shape
    assert np.allclose(result, expected)
    if isinstance(expected, u.Quantity):
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(expected.unit)


@pytest.mark.parametrize(
    argnames="spd,wavelength,axis",
    argvalues=[
        (
            rng.uniform(size=(16, 17, 101)),
            np.linspace(380, 780, num=100) * u.nm,
            -1,
        ),
        (
            rng.uniform(size=(101, 16, 17)),
            np.linspace(380, 780, num=17) * u.nm,
            0,
        ),
        (
            rng.uniform(size=(16, 17, 101)),
            np.linspace(380, 780, num=101) * u.nm,
            3,
        ),
    ],
)
def test_XYZcie1931_from_spd_invalid(
    spd: np.ndarray,
    wavelength: u.Quantity,
    axis: int,
):
    with pytest.raises(ValueError):
        colorsynth.XYZcie1931_from_spd(
            spd=spd,
            wavelength=wavelength,
            axis=axis,
        )


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
    assert np.allclose(np.take(result, 1, axis=axis).max(), 1)


@pytest.mark.parametrize(
    argnames="XYZ,axis,axis_max,axis_check",
    argvalues=[
        (rng.uniform(size=(5, 64, 64, 3)), -1, (1, 2), (1, 2)),
        (rng.uniform(size=(5, 64, 64, 3)), -1, (-3, -2), (1, 2)),
        (rng.uniform(size=(64, 64, 3)), -1, 0, 0),
        (rng.uniform(size=(3, 5, 64, 64)), 0, (2, 3), (1, 2)),
    ],
)
def test_XYZ_normalized_axis_max(
    XYZ: np.ndarray,
    axis: int,
    axis_max: None | int | tuple[int, ...],
    axis_check: int | tuple[int, ...],
):
    result = colorsynth.XYZ_normalized(XYZ, axis=axis, axis_max=axis_max)
    Y = np.take(result, 1, axis=axis)
    assert np.allclose(Y.max(axis=axis_check), 1)


def test_XYZ_normalized_axis_max_invalid():
    XYZ = np.random.default_rng(0).uniform(size=(64, 64, 3))
    with pytest.raises(ValueError):
        colorsynth.XYZ_normalized(XYZ, axis=-1, axis_max=(0, -1))


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
        rng.uniform(size=(101,)),
        rng.uniform(size=(64, 64, 101)),
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
        rng.uniform(size=(101,)),
        rng.uniform(size=(64, 64, 101)),
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
    argnames="shape_spd,axis",
    argvalues=[
        ((51,), -1),
        ((8, 51), -1),
        ((8, 9, 51), -1),
        ((8, 9, 10, 51), -1),
        ((51, 8), 0),
        ((51, 8, 9), 0),
    ],
)
@pytest.mark.parametrize(argnames="num_intensity", argvalues=[11, 101])
def test_colorbar_shape(
    shape_spd: tuple[int, ...],
    axis: int,
    num_intensity: int,
):
    """
    The colorbar must have a consistent shape for any number of dimensions
    of `spd`, with intensity along the first axis and wavelength along the
    second axis by default.
    """
    num_wavelength = shape_spd[axis]
    spd = np.random.default_rng(0).uniform(size=shape_spd)
    wavelength = np.linspace(380, 780, num=num_wavelength) * u.nm
    shape_wavelength = [1] * len(shape_spd)
    shape_wavelength[axis] = num_wavelength
    wavelength = wavelength.reshape(shape_wavelength)

    intensity, wavelength2, RGB = colorsynth.colorbar(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
        num_intensity=num_intensity,
    )

    shape_expected = (num_intensity, num_wavelength)
    assert intensity.shape == shape_expected
    assert wavelength2.shape == shape_expected
    assert RGB.shape == shape_expected + (3,)

    if spd.ndim > 1:
        # If `spd` is one-dimensional there are no axes orthogonal to the
        # wavelength axis, so the automatic normalization bounds are equal
        # to each other and the normalization divides by zero.
        assert np.all(np.isfinite(RGB))


def test_colorbar_duplicate_wavelength():
    """
    A repeated wavelength must give the same colorbar as a wavelength grid
    where the repeat has been perturbed by a negligible amount.
    """
    spd = np.array([0.1, 0.5, 0.5, 0.9])
    wavelength = [400, 500, 500, 600] * u.nm
    wavelength_eps = [400, 500, 500.001, 600] * u.nm
    result = colorsynth.colorbar(
        spd=spd,
        wavelength=wavelength,
        spd_min=0.0,
        spd_max=1.0,
    )
    result_eps = colorsynth.colorbar(
        spd=spd,
        wavelength=wavelength_eps,
        spd_min=0.0,
        spd_max=1.0,
    )
    assert np.all(np.isfinite(result[2]))
    assert np.allclose(result[2], result_eps[2], atol=1e-4)


@pytest.mark.parametrize(
    argnames="spd,wavelength,axis",
    argvalues=[
        (rng.uniform(size=(101,)), np.linspace(380, 780, num=101) * u.nm, -1),
        (rng.uniform(size=(16, 17, 101)), np.linspace(380, 780, num=101) * u.nm, -1),
        (
            rng.uniform(size=(101, 16, 17)),
            np.linspace(380, 780, num=101)[:, np.newaxis, np.newaxis] * u.nm,
            0,
        ),
        (rng.uniform(size=(16, 17, 101)) * u.photon, None, -1),
    ],
)
def test_rgb_and_colorbar_equivalence(
    spd: np.ndarray,
    wavelength: None | u.Quantity,
    axis: int,
):
    """
    Sharing the normalization bounds must not change the result of calling
    :func:`colorsynth.rgb` and :func:`colorsynth.colorbar` independently.
    """
    RGB, cbar = colorsynth.rgb_and_colorbar(
        spd=spd,
        wavelength=wavelength,
        axis=axis,
    )

    RGB_expected = colorsynth.rgb(spd=spd, wavelength=wavelength, axis=axis)
    cbar_expected = colorsynth.colorbar(spd=spd, wavelength=wavelength, axis=axis)

    assert np.array_equal(RGB, RGB_expected, equal_nan=True)
    for r, e in zip(cbar, cbar_expected):
        assert np.array_equal(r, e, equal_nan=True)


@pytest.mark.parametrize(
    argnames="spd",
    argvalues=[
        rng.uniform(size=(101,)),
        rng.uniform(size=(64, 64, 101)),
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
