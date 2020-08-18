import numpy as np
from autoconf import conf
from autocti.mask.mask import Mask, SettingsMask


class SettingsCIMask:
    def __init__(self):

        pass


class CIMask(Mask):
    @classmethod
    def masked_parallel_front_edge_from_ci_frame(cls, ci_frame, rows, invert=False):

        front_edge_regions = ci_frame.parallel_front_edge_regions(rows=rows)
        mask = np.full(ci_frame.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_parallel_trails_from_ci_frame(cls, ci_frame, rows, invert=False):

        trails_regions = ci_frame.parallel_trails_regions(rows=rows)
        mask = np.full(ci_frame.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_serial_front_edge_from_ci_frame(cls, ci_frame, columns, invert=False):

        front_edge_regions = ci_frame.serial_front_edge_regions(columns=columns)
        mask = np.full(ci_frame.shape_2d, False)

        for region in front_edge_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))

    @classmethod
    def masked_serial_trails_from_ci_frame(cls, ci_frame, columns, invert=False):

        trails_regions = ci_frame.serial_trails_regions(columns=columns)
        mask = np.full(ci_frame.shape_2d, False)

        for region in trails_regions:
            mask[region.y0 : region.y1, region.x0 : region.x1] = True

        if invert:
            mask = np.invert(mask)

        return CIMask(mask=mask.astype("bool"))
