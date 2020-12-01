""" 
Find warm pixels in an image from the Hubble Space Telescope (HST) Advanced 
Camera for Surveys (ACS) instrument.

A small patch of the image is plotted with the warm pixels marked with red Xs.
"""

import numpy as np
import pytest
import os
import matplotlib.pyplot as plt

from autocti.data.pixel_lines import PixelLine, PixelLineCollection
from autocti.model.warm_pixels import find_warm_pixels
from autoarray.instruments import acs

# Path to this file
path = os.path.dirname(os.path.realpath(__file__))

# Load the HST ACS dataset
name = "acs/jc0a01h8q_raw"
frame = acs.ImageACS.from_fits(file_path=f"{path}/{name}.fits", quadrant_letter="A")


def prescan_fitted_bias_column(prescan, n_rows=2048, n_rows_ov=20):
    """ 
    Generate a bias column to be subtracted from the main image by doing a
    least squares fit to the serial prescan region.
    
    e.g. image -= prescan_fitted_bias_column(image[18:24])
    
    See Anton & Rorres (2013), S9.3, p460.
    
    Parameters
    ----------
    prescan : [[float]]
        The serial prescan part of the image. Should usually cover the full 
        number of rows but may skip the first few columns of the prescan to 
        avoid trails.
        
    n_rows : int
        The number of rows in the image, exculding overscan.
        
    n_rows_ov : int, int
        The number of overscan rows in the image.
        
    Returns
    -------
    bias_column : [float]
        The fitted bias to be subtracted from all image columns. 
    """
    n_columns_fit = prescan.shape[1]

    # Flatten the multiple fitting columns to a long 1D array
    # y = [y_1_1, y_2_1, ..., y_nrow_1, y_1_2, y_2_2, ..., y_nrow_ncolfit]
    y = prescan[:-n_rows_ov].T.flatten()
    # x = [1, 2, ..., nrow, 1, ..., nrow, 1, ..., nrow, ...]
    x = np.tile(np.arange(n_rows), n_columns_fit)

    # M = [[1, 1, ..., 1], [x_1, x_2, ..., x_n]].T
    M = np.array([np.ones(n_rows * n_columns_fit), x]).T

    # Best-fit values for y = M v
    v = np.dot(np.linalg.inv(np.dot(M.T, M)), np.dot(M.T, y))

    # Map to full image size for easy subtraction
    bias_column = v[0] + v[1] * np.arange(n_rows + n_rows_ov)

    print("# fitted bias v =", v)
    # plt.figure()
    # pixels = np.arange(n_rows + n_rows_ov)
    # for i in range(n_columns_fit):
    #     plt.scatter(pixels, prescan[:, i])
    # plt.plot(pixels, bias_column)
    # plt.show()

    return np.transpose([bias_column])


# Load and subtract the bias image
bias_name = "acs/25b1734qj_bia"
bias = acs.FrameACS.from_fits(file_path=f"{path}/{bias_name}.fits", quadrant_letter="A")
print(np.amin(bias), np.mean(bias), np.median(bias), np.amax(bias))
frame -= bias

# Subtract from all columns the fitted prescan bias
frame -= prescan_fitted_bias_column(frame[:, 18:24])

# Extract an example patch of the full image
row_start, row_end, column_start, column_end = -300, -100, -300, -100
frame = frame[row_start:row_end, column_start:column_end]
frame.mask = frame.mask[row_start:row_end, column_start:column_end]

# Find the warm pixel trails and store in a line collection object
warm_pixels = PixelLineCollection(lines=find_warm_pixels(image=frame))

print("Found %d warm pixels" % warm_pixels.n_lines)

# Plot the image and the found warm pixels
plt.figure()
im = plt.imshow(X=frame, aspect="equal", vmin=0, vmax=500)
plt.scatter(
    warm_pixels.locations[:, 1],
    warm_pixels.locations[:, 0],
    c="r",
    marker="x",
    s=4,
    linewidth=0.2,
)
plt.colorbar(im)
plt.axis("off")
plt.savefig(f"{path}/find_warm_pixels.png", dpi=400)
plt.close()
print(f"Saved {path}/find_warm_pixels.png")
