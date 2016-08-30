import numpy as np


class Otsu():
    def __init__(self):
        pass

    def histogram(image, nbins=256):
        """Return histogram of image.

        Unlike `numpy.histogram`, this function returns the centers of bins and
        does not rebin integer arrays. For integer arrays, each integer value
        has its own bin, which improves speed and intensity-resolution.

        The histogram is computed on the flattened image: for color images, the
        function should be used separately on each channel to obtain a
        histogram for each color channel.

        Parameters
        ----------
        image : array
            Input image.
        nbins : int
            Number of bins used to calculate histogram. This value is ignored for
            integer arrays.

        Returns
        -------
        hist : array
            The values of the histogram.
        bin_centers : array
            The values at the center of the bins.

        See Also
        --------
        cumulative_distribution

        Examples
        --------
        >>> from skimage import data, exposure, img_as_float
        >>> image = img_as_float(data.camera())
        >>> np.histogram(image, bins=2)
        (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
        >>> exposure.histogram(image, nbins=2)
        (array([107432, 154712]), array([ 0.25,  0.75]))
        """
        sh = image.shape
        if len(sh) == 3 and sh[-1] < 4:
            warn("This might be a color image. The histogram will be "
                 "computed on the flattened image. You can instead "
                 "apply this function to each color channel.")

        # For integer types, histogramming with bincount is more efficient.
        if np.issubdtype(image.dtype, np.integer):
            offset = 0
            image_min = np.min(image)
            if image_min < 0:
                offset = image_min
                image_range = np.max(image).astype(np.int64) - image_min
                # get smallest dtype that can hold both minimum and offset maximum
                offset_dtype = np.promote_types(np.min_scalar_type(image_range),
                                                np.min_scalar_type(image_min))
                if image.dtype != offset_dtype:
                    # prevent overflow errors when offsetting
                    image = image.astype(offset_dtype)
                image = image - offset
            hist = np.bincount(image.ravel())
            bin_centers = np.arange(len(hist)) + offset

            # clip histogram to start with a non-zero bin
            idx = np.nonzero(hist)[0][0]
            return hist[idx:], bin_centers[idx:]
        else:
            hist, bin_edges = np.histogram(image.flat, bins=nbins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
            return hist, bin_centers


    def threshold_otsu(image, nbins=256):
        """Return threshold value based on Otsu's method.

        Parameters

        ----------
        image : (N, M) ndarray
            Grayscale input image.
        nbins : int, optional
            Number of bins used to calculate histogram. This value is ignored for
            integer arrays.

        Returns
        -------
        threshold : float
            Upper threshold value. All pixels with an intensity higher than
            this value are assumed to be foreground.

        Raises
        ------
        ValueError
             If `image` only contains a single grayscale value.

        Examples
        --------
        >>> from skimage.data import camera
        >>> image = camera()
        >>> thresh = threshold_otsu(image)
        >>> binary = image <= thresh

        Notes
        -----
        The input image must be grayscale.
        """
        if len(image.shape) > 2 and image.shape[-1] in (3, 4):
            msg = "threshold_otsu is expected to work correctly only for " \
                  "grayscale images; image shape {0} looks like an RGB image"
            warn(msg.format(image.shape))

        # Check if the image is multi-colored or not
        if image.min() == image.max():
            raise ValueError("threshold_otsu is expected to work with images "
                             "having more than one color. The input image seems "
                             "to have just one color {0}.".format(image.min()))

        hist, bin_centers = histogram(image.ravel(), nbins)
        hist = hist.astype(float)

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]

        # Clip ends to align class 1 and class 2 variables:
        # The last value of `weight1`/`mean1` should pair with zero values in
        # `weight2`/`mean2`, which do not exist.
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        idx = np.argmax(variance12)
        threshold = bin_centers[:-1][idx]
        return threshold

if __name__ == "__main__":
    pass
