import numpy as np
import pytest

from autocti import mock

### Arctic ###


@pytest.fixture(name="trap_0")
def make_trap_0():
    return mock.make_trap_0()


@pytest.fixture(name="trap_1")
def make_trap_1():
    return mock.make_trap_1()


@pytest.fixture(name="traps_x1")
def make_traps_x1():
    return mock.make_traps_x1()


@pytest.fixture(name="traps_x2")
def make_traps_x2():
    return mock.make_traps_x2()


@pytest.fixture(name="ccd")
def make_ccd():
    return mock.make_ccd()


@pytest.fixture(name="ccd_complex")
def make_ccd_complex():
    return mock.make_ccd_complex()


@pytest.fixture(name="parallel_clocker")
def make_parallel_clocker():
    return mock.make_parallel_clocker()


@pytest.fixture(name="serial_clocker")
def make_serial_clocker():
    return mock.make_serial_clocker()


### MASK ###


@pytest.fixture(name="mask_7x7")
def make_mask_7x7():
    return mock.make_mask_7x7()


### FRAMES ###


@pytest.fixture(name="scans_7x7")
def make_scans_7x7():
    return mock.make_scans_7x7()


@pytest.fixture(name="image_7x7")
def make_image_7x7():
    return mock.make_image_7x7()


@pytest.fixture(name="noise_map_7x7")
def make_noise_map_7x7():
    return mock.make_noise_map_7x7()


### IMAGING ###


@pytest.fixture(name="imaging_7x7")
def make_imaging_7x7():
    return mock.make_imaging_7x7()


### CHARGE INJECTION FRAMES ###


@pytest.fixture(name="ci_pattern_7x7")
def make_ci_pattern_7x7():
    return mock.make_ci_pattern_7x7()


@pytest.fixture(name="ci_image_7x7")
def make_ci_image_7x7():
    return mock.make_ci_image_7x7()


@pytest.fixture(name="ci_noise_map_7x7")
def make_ci_noise_map_7x7():
    return mock.make_ci_noise_map_7x7()


@pytest.fixture(name="ci_pre_cti_7x7")
def make_ci_pre_cti_7x7():
    return mock.make_ci_pre_cti_7x7()


@pytest.fixture(name="ci_cosmic_ray_map_7x7")
def make_ci_cosmic_ray_map_7x7():
    return mock.make_ci_cosmic_ray_map_7x7()


@pytest.fixture(name="ci_noise_scaling_maps_7x7")
def make_ci_noise_scaling_maps_7x7():

    return mock.make_ci_noise_scaling_maps_7x7()


### CHARGE INJECTION IMAGING ###


@pytest.fixture(name="ci_imaging_7x7")
def make_ci_imaging_7x7():

    return mock.make_ci_imaging_7x7()


@pytest.fixture(name="masked_ci_imaging_7x7")
def make_masked_ci_imaging_7x7():
    return mock.make_masked_ci_imaging_7x7()


### CHARGE INJECTION FITS ###


@pytest.fixture(name="hyper_noise_scalars")
def make_hyper_noise_scalars():
    return mock.make_hyper_noise_scalars()


@pytest.fixture(name="ci_fit_7x7")
def make_ci_fit_7x7():
    return mock.make_ci_fit_7x7()


# ### PHASES ###

from autofit.mapper.model import ModelInstance


@pytest.fixture(name="samples_with_result")
def make_samples_with_result():
    return mock.make_samples_with_result()


@pytest.fixture(name="phase_dataset_7x7")
def make_phase_data():
    return mock.make_phase_data()


@pytest.fixture(name="phase_ci_imaging_7x7")
def make_phase_ci_imaging_7x7():
    return mock.make_phase_ci_imaging_7x7()
