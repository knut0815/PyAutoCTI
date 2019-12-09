import numpy as np
import pytest

import autoarray as aa
import autocti as ac

from test_autocti.mock.mock import (
    MockGeometry,
    MockPattern,
    MockCIFrame,
    MockCIPreCTI,
    MockParams,
    MockSettings,
)


@pytest.fixture(name="ci_data_masked")
def make_ci_data_fit():

    ci_image = 1.0 * np.ones((2, 2))
    ci_noise_map = 2.0 * np.ones((2, 2))

    ci_pre_cti = MockCIPreCTI(
        frame_geometry=MockGeometry(),
        ci_pattern=MockPattern(),
        array=3.0 * np.ones((2, 2)),
        value=3.0,
    )
    ci_mask = np.ma.zeros((2, 2))

    return ac.CIMaskedImaging(
        ci_frame=MockCIFrame(value=3.0),
        ci_pattern=MockPattern(),
        image=ci_image,
        noise_map=ci_noise_map,
        ci_pre_cti=ci_pre_cti,
        mask=ci_mask,
    )


@pytest.fixture(name="ci_data_hyper_fit")
def make_ci_datas_hyper_fit(ci_data_masked):
    ci_noise_scalings = [
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ]

    return ac.CIMaskedImaging(
        ci_frame=MockCIFrame(value=3.0),
        ci_pattern=MockPattern(),
        image=ci_data_masked.image,
        noise_map=ci_data_masked.noise_map,
        ci_pre_cti=ci_data_masked.ci_pre_cti,
        mask=ci_data_masked.mask,
        noise_scaling_maps=ci_noise_scalings,
    )


class TestCIFit:
    def test__fit_quantities_same_as_calculated_individually(self, ci_data_masked):

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_masked,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        residual_map = aa.util.fit.residual_map_from_data_mask_and_model_data(
            data=ci_data_masked.image,
            mask=ci_data_masked.mask,
            model_data=ci_data_masked.ci_pre_cti,
        )

        assert (fit.residual_map == residual_map).all()

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map,
            noise_map=ci_data_masked.noise_map,
            mask=ci_data_masked.mask,
        )

        assert (fit.chi_squared_map == chi_squared_map).all()

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map, mask=ci_data_masked.mask
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map_and_mask(
            noise_map=ci_data_masked.noise_map, mask=ci_data_masked.mask
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_masked,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        assert fit.likelihood == likelihood
        assert fit.figure_of_merit == fit.likelihood

    def test__image_and_ci_post_cti_the_same__noise_scaling_all_0s__likelihood_is_noise_normalization(
        self
    ):
        ci_image = 10.0 * np.ones((2, 2))
        ci_noise_map = 2.0 * np.ones((2, 2))
        ci_mask = np.ma.zeros((2, 2))
        ci_pre_cti = 10.0 * np.ones((2, 2))
        ci_noise_scalings = [np.array([[0.0, 0.0], [0.0, 0.0]])]

        ci_data_hyper_fit = ac.CIMaskedImaging(
            ci_frame=MockCIFrame(value=10.0),
            ci_pattern=MockPattern(),
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            mask=ci_mask,
            noise_scaling_maps=ci_noise_scalings,
        )

        hyper_noise_scalar = ac.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_hyper_fit,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__image_and_post_cti_different__noise_scaling_all_0s__likelihood_chi_squared_and_noise_normalization(
        self
    ):
        ci_image = 9.0 * np.ones((2, 2))
        ci_noise_map = 2.0 * np.ones((2, 2))
        ci_mask = np.ma.zeros((2, 2))
        ci_pre_cti = 10.0 * np.ones((2, 2))
        ci_noise_scalings = [np.array([[0.0, 0.0], [0.0, 0.0]])]

        ci_data_hyper_fit = ac.CIMaskedImaging(
            ci_frame=MockCIFrame(value=10.0),
            ci_pattern=MockPattern(),
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            mask=ci_mask,
            noise_scaling_maps=ci_noise_scalings,
        )

        hyper_noise_scalar = ac.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_hyper_fit,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 4.0 * ((1.0 / 2.0) ** 2.0)
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.likelihood == pytest.approx(
            -0.5 * (chi_squared + noise_normalization), 1e-4
        )

    def test__image_and_ci_post_cti_the_same__noise_scaling_non_0s__likelihood_is_noise_normalization(
        self
    ):
        ci_image = 10.0 * np.ones((2, 2))
        ci_noise_map = 2.0 * np.ones((2, 2))
        ci_mask = np.ma.zeros((2, 2))
        ci_pre_cti = MockCIPreCTI(
            frame_geometry=MockGeometry(),
            ci_pattern=MockPattern(),
            array=10.0 * np.ones((2, 2)),
            value=10.0,
        )

        ci_noise_scalings = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

        ci_data_hyper_fit = ac.CIMaskedImaging(
            ci_frame=MockCIFrame(value=10.0),
            ci_pattern=MockPattern(),
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            mask=ci_mask,
            noise_scaling_maps=ci_noise_scalings,
        )

        hyper_noise_scalar = ac.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_hyper_fit,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (2.0 + 1.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 3.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)
        )

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self
    ):
        ci_image = 10.0 * np.ones((2, 2))
        ci_noise_map = 3.0 * np.ones((2, 2))
        ci_mask = np.ma.zeros((2, 2))
        ci_pre_cti = MockCIPreCTI(
            frame_geometry=MockGeometry(),
            ci_pattern=MockPattern(),
            array=10.0 * np.ones((2, 2)),
            value=10.0,
        )

        ci_noise_scalings = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[5.0, 6.0], [7.0, 8.0]]),
        ]

        ci_data_hyper_fit = ac.CIMaskedImaging(
            ci_frame=MockCIFrame(value=10.0),
            ci_pattern=MockPattern(),
            image=ci_image,
            noise_map=ci_noise_map,
            ci_pre_cti=ci_pre_cti,
            mask=ci_mask,
            noise_scaling_maps=ci_noise_scalings,
        )

        hyper_noise_scalar_0 = ac.CIHyperNoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ac.CIHyperNoiseScalar(scale_factor=2.0)

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_hyper_fit,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (3.0 + 1.0 + 10.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 2.0 + 12.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 3.0 + 14.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 4.0 + 16.0) ** 2.0)
        )

        assert fit.likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__all_quantities_are_same_as_calculated_individually(
        self, ci_data_hyper_fit
    ):
        hyper_noise_scalar_0 = ac.CIHyperNoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ac.CIHyperNoiseScalar(scale_factor=2.0)

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_hyper_fit,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        hyper_noise_map = ac.hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_map=ci_data_hyper_fit.noise_map,
            noise_scaling_maps=ci_data_hyper_fit.noise_scaling_maps,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        assert (hyper_noise_map == fit.noise_map).all()

        residual_map = aa.util.fit.residual_map_from_data_mask_and_model_data(
            data=ci_data_hyper_fit.image,
            mask=ci_data_hyper_fit.mask,
            model_data=ci_data_hyper_fit.ci_pre_cti,
        )

        assert (residual_map == fit.residual_map).all()

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map,
            noise_map=hyper_noise_map,
            mask=ci_data_hyper_fit.mask,
        )

        assert (chi_squared_map == fit.chi_squared_map).all()

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map, mask=ci_data_hyper_fit.mask
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map_and_mask(
            noise_map=hyper_noise_map, mask=ci_data_hyper_fit.mask
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert (likelihood == fit.likelihood).all()


class TestHyperNoiseMap:
    def test__image_and_pre_cti_not_identical__noise_scalings_ares__no_noise_map_scaling(
        self
    ):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling_maps = [np.array([[0.0, 0.0], [0.0, 0.0]])]
        hyper_noise_scalars = [ac.CIHyperNoiseScalar(scale_factor=1.0)]

        noise_map = ac.hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__factor_is__no_noise_map_scaling(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling_maps = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise_scalars = [ac.CIHyperNoiseScalar(scale_factor=0.0)]

        noise_map = ac.hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_by_noise_map(self):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling_maps = [np.array([[1.0, 2.0], [3.0, 4.0]])]
        hyper_noise_scalars = [ac.CIHyperNoiseScalar(scale_factor=1.0)]

        noise_map = ac.hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[3.0, 4.0], [5.0, 6.0]]))).all()

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self
    ):
        noise_map = 2.0 * np.ones((2, 2))
        noise_scaling_maps = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ]
        hyper_noise_scalars = [
            ac.CIHyperNoiseScalar(scale_factor=1.0),
            ac.CIHyperNoiseScalar(scale_factor=2.0),
        ]

        noise_map = ac.hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[5.0, 8.0], [11.0, 14.0]]))).all()


class MockCIDataFit(object):
    def __init__(self, image, noise_map, ci_pre_cti, mask, ci_pattern, ci_frame):
        self.image = image
        self.noise_map = noise_map
        self.ci_pre_cti = ci_pre_cti
        self.mask = mask
        self.ci_pattern = ci_pattern
        self.ci_frame = ci_frame
        self.is_hyper_data = False

    @property
    def chinj(self):
        return MockCIFrame(value=1.0)


@pytest.fixture(name="ci_data_fit_1")
def make_ci_data_fit_1():
    ci_image = np.array([[np.sqrt(1.0), np.sqrt(2.0)], [np.sqrt(3.0), np.sqrt(4.0)]])
    ci_noise_map = 1.0 * np.ones((2, 2))
    ci_pattern = MockPattern(regions=[ac.Region(region=[0, 1, 0, 1])])

    ci_pre_cti = MockCIPreCTI(
        frame_geometry=MockGeometry(),
        ci_pattern=ci_pattern,
        array=0.0 * np.ones((2, 2)),
        value=0.0,
    )
    ci_mask = np.ma.zeros((2, 2))

    return MockCIDataFit(
        ci_frame=MockCIFrame(value=0.0),
        ci_pattern=ci_pattern,
        image=ci_image,
        noise_map=ci_noise_map,
        ci_pre_cti=ci_pre_cti,
        mask=ci_mask,
    )


class TestChiSquaredMapsOfRegions:
    def test__chi_squared_map_of_ci_regions__extracts_correctly_from_chi_squard_map(
        self, ci_data_fit_1
    ):
        # For the mock object MockCIFrame, the ci_regions_from_array function extracts the arrays entries
        # [0:2,0]

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_fit_1,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        assert (fit.chi_squared_map_of_ci_regions == fit.chi_squared_map[0:2, 0]).all()

    def test__chi_squared_map_of_parallel_non_ci_regions__extracts_correctly_from_chi_squard_map(
        self, ci_data_fit_1
    ):
        # For the mock object MockCIFrame, the parallel_non_ci_regions_frame_from_frame function extracts the arrays
        # entries [0:2,1]

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_fit_1,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        assert (
            fit.chi_squared_map_of_parallel_trails == fit.chi_squared_map[0:2, 1]
        ).all()

    def test__chi_squared_map_of_serial_trails__extracts_correctly_from_chi_squard_map(
        self, ci_data_fit_1
    ):
        # For the mock object MockCIFrame, the parallel_non_ci_regions_frame_from_frame function extracts the arrays
        # entries [0, 0:2]

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_fit_1,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        assert (
            fit.chi_squared_map_of_serial_trails == fit.chi_squared_map[0, 0:2]
        ).all()

    def test__chi_squared_map_of_overscan_above_serial_trails__extracts_correctly_from_chi_squard_map(
        self, ci_data_fit_1
    ):
        # For the mock object MockCIFrame, the parallel_non_ci_regions_frame_from_frame function extracts the arrays
        # entries [0, 0:2]

        fit = ac.CIImagingFit(
            ci_masked_imaging=ci_data_fit_1,
            cti_params=MockParams(),
            cti_settings=MockSettings(),
        )

        assert (
            fit.chi_squared_map_of_serial_overscan_above_trails
            == fit.chi_squared_map[1, 0:2]
        ).all()
