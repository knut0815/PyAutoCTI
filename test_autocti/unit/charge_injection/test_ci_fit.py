import numpy as np
import pytest
from autocti import charge_injection as ci
from autocti import util
from autocti.charge_injection.ci_fit import (
    hyper_noise_map_from_noise_map_and_noise_scalings,
)


class TestCIFit:
    def test__fit_quantities_same_as_calculated_individually(
        self, masked_ci_imaging_7x7, ci_pattern_7x7
    ):

        ci_post_cti = ci.CIFrame.ones(
            shape_2d=masked_ci_imaging_7x7.image.shape_2d, ci_pattern=ci_pattern_7x7
        )

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7, ci_post_cti=ci_post_cti
        )

        residual_map = util.fit.residual_map_from_data_mask_and_model_data(
            data=masked_ci_imaging_7x7.image,
            mask=masked_ci_imaging_7x7.mask,
            model_data=ci_post_cti,
        )

        assert (fit.residual_map == residual_map).all()

        chi_squared_map = util.fit.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map,
            noise_map=masked_ci_imaging_7x7.noise_map,
            mask=masked_ci_imaging_7x7.mask,
        )

        assert (fit.chi_squared_map == chi_squared_map).all()

        chi_squared = util.fit.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map, mask=masked_ci_imaging_7x7.mask
        )

        noise_normalization = util.fit.noise_normalization_from_noise_map_and_mask(
            noise_map=masked_ci_imaging_7x7.noise_map, mask=masked_ci_imaging_7x7.mask
        )

        log_likelihood = util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert fit.log_likelihood == log_likelihood

    def test__image_and_ci_post_cti_the_same__noise_scaling_all_0s__likelihood_is_noise_normalization(
        self, ci_pattern_7x7
    ):

        image = 10.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7)
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        ci_pre_cti = 10.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7)

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))
        noise_scaling_maps = [
            ci.CIFrame.zeros(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7),
            ci.CIFrame.zeros(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7),
        ]

        masked_imaging = ci.MaskedCIImaging(
            ci_imaging=imaging, mask=mask, noise_scaling_maps=noise_scaling_maps
        )

        hyper_noise_scalar = ci.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_imaging,
            ci_post_cti=ci_pre_cti,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__image_and_post_cti_different__noise_scaling_all_0s__likelihood_chi_squared_and_noise_normalization(
        self, ci_pattern_7x7
    ):
        image = ci.CIFrame.full(
            fill_value=9.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        ci_pre_cti = ci.CIFrame.full(
            fill_value=10.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))
        noise_scaling_maps = [
            ci.CIFrame.manual(array=[[0.0, 0.0], [0.0, 0.0]], ci_pattern=ci_pattern_7x7)
        ]

        masked_imaging = ci.MaskedCIImaging(
            ci_imaging=imaging, mask=mask, noise_scaling_maps=noise_scaling_maps
        )

        hyper_noise_scalar = ci.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_imaging,
            ci_post_cti=ci_pre_cti,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 4.0 * ((1.0 / 2.0) ** 2.0)
        noise_normalization = 4.0 * np.log(2 * np.pi * 4.0)

        assert fit.log_likelihood == pytest.approx(
            -0.5 * (chi_squared + noise_normalization), 1e-4
        )

    def test__image_and_ci_post_cti_the_same__noise_scaling_non_0s__likelihood_is_noise_normalization(
        self, ci_pattern_7x7
    ):
        image = 10.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7)
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        ci_pre_cti = ci.CIFrame.full(
            fill_value=10.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        noise_scaling_maps = [
            ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7
            ),
            ci.CIFrame.manual(
                array=[[5.0, 6.0], [7.0, 8.0]], ci_pattern=ci_pattern_7x7
            ),
        ]

        masked_imaging = ci.MaskedCIImaging(
            ci_imaging=imaging, mask=mask, noise_scaling_maps=noise_scaling_maps
        )

        hyper_noise_scalar = ci.CIHyperNoiseScalar(scale_factor=1.0)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_imaging,
            ci_post_cti=ci_pre_cti,
            hyper_noise_scalars=[hyper_noise_scalar],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (2.0 + 1.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 2.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 3.0) ** 2.0)
            + np.log(2 * np.pi * (2.0 + 4.0) ** 2.0)
        )

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self, ci_pattern_7x7
    ):
        image = 10.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7)
        noise_map = 3.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=ci_pattern_7x7)
        ci_pre_cti = ci.CIFrame.full(
            fill_value=10.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        noise_scaling_maps = [
            ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7
            ),
            ci.CIFrame.manual(
                array=[[5.0, 6.0], [7.0, 8.0]], ci_pattern=ci_pattern_7x7
            ),
        ]

        masked_imaging = ci.MaskedCIImaging(
            ci_imaging=imaging, mask=mask, noise_scaling_maps=noise_scaling_maps
        )

        hyper_noise_scalar_0 = ci.CIHyperNoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ci.CIHyperNoiseScalar(scale_factor=2.0)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_imaging,
            ci_post_cti=ci_pre_cti,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        chi_squared = 0
        noise_normalization = (
            np.log(2 * np.pi * (3.0 + 1.0 + 10.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 2.0 + 12.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 3.0 + 14.0) ** 2.0)
            + np.log(2 * np.pi * (3.0 + 4.0 + 16.0) ** 2.0)
        )

        assert fit.log_likelihood == -0.5 * (chi_squared + noise_normalization)

    def test__all_quantities_are_same_as_calculated_individually(
        self, masked_ci_imaging_7x7, ci_pattern_7x7
    ):
        hyper_noise_scalar_0 = ci.CIHyperNoiseScalar(scale_factor=1.0)
        hyper_noise_scalar_1 = ci.CIHyperNoiseScalar(scale_factor=2.0)

        fit = ci.CIFitImaging(
            masked_ci_imaging=masked_ci_imaging_7x7,
            ci_post_cti=masked_ci_imaging_7x7.ci_pre_cti,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        hyper_noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            noise_map=masked_ci_imaging_7x7.noise_map,
            noise_scaling_maps=masked_ci_imaging_7x7.noise_scaling_maps,
            hyper_noise_scalars=[hyper_noise_scalar_0, hyper_noise_scalar_1],
        )

        assert (hyper_noise_map == fit.noise_map).all()

        residual_map = util.fit.residual_map_from_data_mask_and_model_data(
            data=masked_ci_imaging_7x7.image,
            mask=masked_ci_imaging_7x7.mask,
            model_data=masked_ci_imaging_7x7.ci_pre_cti,
        )

        assert (residual_map == fit.residual_map).all()

        chi_squared_map = util.fit.chi_squared_map_from_residual_map_noise_map_and_mask(
            residual_map=residual_map,
            noise_map=hyper_noise_map,
            mask=masked_ci_imaging_7x7.mask,
        )

        assert (chi_squared_map == fit.chi_squared_map).all()

        chi_squared = util.fit.chi_squared_from_chi_squared_map_and_mask(
            chi_squared_map=chi_squared_map, mask=masked_ci_imaging_7x7.mask
        )

        noise_normalization = util.fit.noise_normalization_from_noise_map_and_mask(
            noise_map=hyper_noise_map, mask=masked_ci_imaging_7x7.mask
        )

        log_likelihood = util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert (log_likelihood == fit.log_likelihood).all()


class TestHyperNoiseMap:
    def test__image_and_pre_cti_not_identical__noise_scaling_maps_ares__no_noise_map_scaling(
        self, ci_pattern_7x7
    ):
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        noise_scaling_maps = [
            ci.CIFrame.manual(array=[[0.0, 0.0], [0.0, 0.0]], ci_pattern=ci_pattern_7x7)
        ]
        hyper_noise_scalars = [ci.CIHyperNoiseScalar(scale_factor=1.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__factor_is__no_noise_map_scaling(
        self, ci_pattern_7x7
    ):
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        noise_scaling_maps = [
            ci.CIFrame.manual(array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7)
        ]
        hyper_noise_scalars = [ci.CIHyperNoiseScalar(scale_factor=0.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[2.0, 2.0], [2.0, 2.0]]))).all()

    def test__image_and_pre_cti_not_identical__chi_sq_is_by_noise_map(
        self, ci_pattern_7x7
    ):
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        noise_scaling_maps = [
            ci.CIFrame.manual(array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7)
        ]
        hyper_noise_scalars = [ci.CIHyperNoiseScalar(scale_factor=1.0)]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[3.0, 4.0], [5.0, 6.0]]))).all()

    def test__x2_noise_map_scaling_and_hyper_params__noise_map_term_comes_out_correct(
        self, ci_pattern_7x7
    ):
        noise_map = ci.CIFrame.full(
            fill_value=2.0, shape_2d=(2, 2), ci_pattern=ci_pattern_7x7
        )
        noise_scaling_maps = [
            ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7
            ),
            ci.CIFrame.manual(
                array=[[1.0, 2.0], [3.0, 4.0]], ci_pattern=ci_pattern_7x7
            ),
        ]
        hyper_noise_scalars = [
            ci.CIHyperNoiseScalar(scale_factor=1.0),
            ci.CIHyperNoiseScalar(scale_factor=2.0),
        ]

        noise_map = hyper_noise_map_from_noise_map_and_noise_scalings(
            hyper_noise_scalars=hyper_noise_scalars,
            noise_map=noise_map,
            noise_scaling_maps=noise_scaling_maps,
        )

        assert (noise_map == (np.array([[5.0, 8.0], [11.0, 14.0]]))).all()


class TestChiSquaredMapsOfRegions:
    def test__chi_squared_map_of_ci_regions__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ci.CIPatternUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=pattern)
        noise_map = ci.CIFrame.ones(shape_2d=(2, 2), ci_pattern=pattern)
        ci_pre_cti = ci.CIFrame.full(
            fill_value=1.0, shape_2d=(2, 2), ci_pattern=pattern
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        masked_imaging = ci.MaskedCIImaging(ci_imaging=imaging, mask=mask)

        fit = ci.CIFitImaging(masked_ci_imaging=masked_imaging, ci_post_cti=ci_pre_cti)

        assert (
            fit.chi_squared_map_of_ci_regions == np.array([[4.0, 0.0], [0.0, 0.0]])
        ).all()

    def test__chi_squared_map_of_parallel_non_ci_regions__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ci.CIPatternUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ci.CIFrame.ones(
            shape_2d=(2, 2),
            ci_pattern=pattern,
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 1, 1, 2),
        )
        noise_map = ci.CIFrame.ones(
            shape_2d=(2, 2),
            ci_pattern=pattern,
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 1, 1, 2),
        )
        ci_pre_cti = ci.CIFrame.full(
            fill_value=1.0,
            shape_2d=(2, 2),
            ci_pattern=pattern,
            serial_prescan=(1, 2, 1, 2),
            serial_overscan=(0, 1, 1, 2),
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        masked_imaging = ci.MaskedCIImaging(ci_imaging=imaging, mask=mask)

        fit = ci.CIFitImaging(masked_ci_imaging=masked_imaging, ci_post_cti=ci_pre_cti)

        assert (
            fit.chi_squared_map_of_parallel_trails == np.array([[0.0, 0.0], [4.0, 0.0]])
        ).all()

    def test__chi_squared_map_of_serial_trails__extracts_correctly_from_chi_squard_map(
        self,
    ):

        pattern = ci.CIPatternUniform(regions=[(0, 2, 0, 1)], normalization=1.0)

        image = 3.0 * ci.CIFrame.ones(
            shape_2d=(2, 2), ci_pattern=pattern, serial_overscan=(1, 2, 0, 2)
        )
        noise_map = ci.CIFrame.ones(
            shape_2d=(2, 2), ci_pattern=pattern, serial_overscan=(1, 2, 0, 2)
        )
        ci_pre_cti = ci.CIFrame.full(
            fill_value=1.0,
            shape_2d=(2, 2),
            ci_pattern=pattern,
            serial_overscan=(1, 2, 0, 2),
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        masked_imaging = ci.MaskedCIImaging(ci_imaging=imaging, mask=mask)

        fit = ci.CIFitImaging(masked_ci_imaging=masked_imaging, ci_post_cti=ci_pre_cti)

        assert (
            fit.chi_squared_map_of_serial_trails == np.array([[0.0, 4.0], [0.0, 4.0]])
        ).all()

    def test__chi_squared_map_of_overscan_above_serial_trails__extracts_correctly_from_chi_squard_map(
        self,
    ):
        pattern = ci.CIPatternUniform(regions=[(0, 1, 0, 1)], normalization=1.0)

        image = 3.0 * ci.CIFrame.ones(
            shape_2d=(2, 2), ci_pattern=pattern, serial_overscan=(0, 2, 1, 2)
        )
        noise_map = ci.CIFrame.ones(
            shape_2d=(2, 2), ci_pattern=pattern, serial_overscan=(0, 2, 1, 2)
        )
        ci_pre_cti = ci.CIFrame.full(
            fill_value=1.0,
            shape_2d=(2, 2),
            ci_pattern=pattern,
            serial_overscan=(0, 2, 1, 2),
        )

        imaging = ci.CIImaging(image=image, noise_map=noise_map, ci_pre_cti=ci_pre_cti)

        mask = np.full(fill_value=False, shape=(2, 2))

        masked_imaging = ci.MaskedCIImaging(ci_imaging=imaging, mask=mask)

        fit = ci.CIFitImaging(masked_ci_imaging=masked_imaging, ci_post_cti=ci_pre_cti)

        assert (
            fit.chi_squared_map_of_serial_overscan_no_trails
            == np.array([[0.0, 0.0], [0.0, 4.0]])
        ).all()
