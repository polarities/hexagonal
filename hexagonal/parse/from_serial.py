from typing import NoReturn
import numpy as np
import warnings
from dataclasses import dataclass
from functools import cache
from abc import ABC
import copy

class HexImageBase(ABC):
    x_coordinates: np.ndarray | None = None
    y_coordinates: np.ndarray | None = None
    image_data: np.ndarray | None = None

    # Metadata
    image_rows_number : int | None = None
    image_columns_number_even : int | None = None
    image_columns_number_odd : int | None = None
    image_columns_offset_even : float | None = None
    image_columns_offset_odd : float | None = None
    image_dx: float | None = None
    image_dy: float | None = None

    @cache
    def _checker(self):
        """Check whether all required objects were given."""
        to_be_checked = (self.x_coordinates,
                         self.y_coordinates,
                         self.image_data,
                         self.image_rows_number,
                         self.image_columns_number_even,
                         self.image_columns_number_odd,
                         self.image_columns_offset_even,
                         self.image_columns_offset_odd)
        if any(x is None for x in to_be_checked):
            raise ValueError("HexImageBase object is not initialized properly.")

    def _set_xystep(self) -> NoReturn:
        """
        Set X and Y step if they are not given entirely or partially given.
        """
        if (self.image_dx is None) and (self.image_dy is None):
            self.image_dx = 1
            self.image_dy = (self.image_dx * np.sqrt(3)) / 2
        elif (self.image_dx is not None) and (self.image_dy is None):
            self.image_dy = (self.image_dx * np.sqrt(3)) / 2
        elif (self.image_dx is None) and (self.image_dy is not None):
            self.image_dx = self.image_dy * (2 / np.sqrt(3))
        else:
            pass

    def _recalculate(self):
        self._checker()
        self._set_xystep()

    def resample2d(self, method='nearest', aspect_ratio_1 = True):
        """
        Resample 2D image from hexagonal image to square grid image. About the interpolation method, please refer to the
         documentation of `scipy.interpolate.griddata`.

        Parameters
        ----------
        method : {‘linear’, ‘nearest’, ‘cubic’}, optional
            Method for interpolation. Default is 'nearest'.
        aspect_ratio_1 : bool, optional
            If True, the pixel width and pixel height ratio will be 1:1. If false, pixel height wise interpolation will
            use the `number_of_rows`, which the data will be more accurate but aspect ratio distorted.

        Notes
        -----
        Image can be distorted. Resampling of Hexagonal to Square grid is not guaranteed to be accurate, especially for
        image data with low resolution, or containing scientific data.

        See Also
        --------
        hexagonal.HexImageBase.square_gridded : Hexagonal image to square grid without resampling the data. Key
        difference is that this function does not interpolate the image data, but return the
        `matplotlib.pyplot.Axes.imshow` plottable 2-dimensional array. This function is recommended for fast visual
        inspection of the image.
        """
        from scipy.interpolate import griddata

        x_axis = np.linspace(np.min(self.x_coordinates),
                             np.max(self.x_coordinates),
                             np.max((self.image_columns_number_even, self.image_columns_number_odd))
                             )
        if aspect_ratio_1 is True:
            interval_aspect_conserving = (np.max(self.x_coordinates) - np.min(self.x_coordinates)) / np.max((self.image_columns_number_even, self.image_columns_number_odd))
            y_axis_rows_number = (np.max(self.y_coordinates) - np.min(self.y_coordinates)) / interval_aspect_conserving
            y_axis_rows_number = int(y_axis_rows_number)
            y_axis = np.linspace(np.min(self.y_coordinates),
                                 np.max(self.y_coordinates),
                                 y_axis_rows_number)
        else:
            y_axis = np.linspace(np.min(self.y_coordinates),
                                 np.max(self.y_coordinates),
                                 self.image_rows_number)

        x, y = np.meshgrid(x_axis, y_axis)

        # Resample image.
        image_data_resampled = griddata((self.x_coordinates, self.y_coordinates), self.image_data, (x, y), method=method)
        return image_data_resampled

    @cache
    def square_gridded(self):
        '''
        Return hexagonal-sampled image to the square-sampled image.

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        This function is recommended for fast visual inspection of the image. The location of the each pixel in the
        square gridded image is not the same as the hexagonal image. Also, you may also expect the height-width ratio
        distortion of the square gridded image. Since hexagonal image x-interval and y-interval are not the same, but
        in this function the x-interval and the y-interval are the same. Therefore, the height-width ratio distortion
        of the square gridded image is significant.

        See Also
        --------
        hexagonal.HexImageBase.resample2d : Resample 2D image from hexagonal image to square grid image. This resamples
        the image data. However, as a result of the resampling, the value of the each pixel cannot be guaranteed.
        '''
        datastream = copy.deepcopy(self.image_data)
        temp_image = list()
        if self.image_columns_number_odd > self.image_columns_number_even:
            for index in range(self.image_rows_number):
                if index % 2 == 1:
                    temp = datastream[:self.image_columns_number_odd]
                    datastream = datastream[self.image_columns_number_odd:]
                    temp_image.append(temp)
                else:
                    temp = datastream[:self.image_columns_number_even]
                    datastream = datastream[self.image_columns_number_even:]
                    try:
                        temp = np.insert(temp, 0, None)
                    except TypeError:
                        temp = np.insert(temp, 0, 0)
                    temp_image.append(temp)
        elif self.image_columns_number_even > self.image_columns_number_odd:
            for index in range(self.image_rows_number):
                if index % 2 == 1:
                    temp = datastream[:self.image_columns_number_odd]
                    datastream = datastream[self.image_columns_number_odd:]
                    try:
                        temp = np.insert(temp, 0, None)
                    except TypeError:
                        temp = np.insert(temp, 0, 0)
                        temp_image.append(temp)
                else:
                    temp = datastream[:self.image_columns_number_even]
                    datastream = datastream[self.image_columns_number_even:]
                    temp_image.append(temp)
        return np.array(temp_image)


    def metadata(self):
        """
        Print 'core' metadata of the image.

        Returns
        -------
        dict
            Dictionary of metadata.
        """
        raise NotImplementedError

    def get_xydata(self):
        """
        Get x, y coordinates and imagedata. Recommended use cases are for plotting by matplotlib manually.
        """
        return self.x_coordinates, self.y_coordinates, self.image_data

    def _test_and_correct_error(self):
        if self._check_hexagonal_dataentry(self.image_columns_number_even,  # Check EBSD size is right.
                                           self.image_columns_number_odd,
                                           self.image_rows_number,
                                           len(self.image_data),
                                           return_value=False):
            pass
        else:  # If data size is not correct, then swap the two columns.
            self.image_columns_number_even, self.image_columns_number_odd = self.image_columns_number_odd, self.image_columns_number_even
            self.image_columns_offset_even, self.image_columns_offset_odd = self.image_columns_offset_odd, self.image_columns_offset_even

            if self._check_hexagonal_dataentry(self.image_columns_number_even,  # Check EBSD size is right.
                                               self.image_columns_number_odd,
                                               self.image_rows_number,
                                               len(self.image_data),
                                               return_value=False):
                warnings.warn(
                    "Error in provided d_sequence array detected. However it looks can be fixed automatically by "
                    "inverting stacking sequence. This can be happened during parsing .ang file.")
            else:
                raise Exception(f"The number of data is incorrect. \n"
                                f"""- Expected: {self._check_hexagonal_dataentry(self.image_columns_number_even,
                                                                                 self.image_columns_number_odd,
                                                                                 self.image_rows_number,
                                                                                 len(self.image_data),
                                                                                 return_value=True)}\n"""
                                f"- Given: {len(self.image_data)} \n")


    @staticmethod
    def _check_hexagonal_dataentry(ncol_even: int,
                                   ncol_odd: int,
                                   nrow: int,
                                   n_dataentry: int,
                                   return_value: False) -> bool:
        """
        Check whether hexagonal data number is correct or not based on the parsed metadata.

        Parameters
        ----------
        ncol_even : int
            NCOLS_EVEN tag data.
        ncol_odd : int
            NCOLS_ODD tag data.
        nrow : int
            NROWS tag data.
        n_dataentry : int
            Number of the data entry of given datafile.
        return_value : int, optional.
            If true, returns calculated value. If false, returns boolean value indicating value is correct or not.

        Returns
        -------
        bool
            True or False if `return_value` is False. Default behavior.
        int
            Number of pixels, if `return_value` is set True.
        """
        remainder = nrow % 2
        neven = nrow // 2 + remainder
        nodd = nrow // 2
        num_pixel = (neven * ncol_even) + (nodd * ncol_odd)
        if return_value is not True:
            return num_pixel == n_dataentry
        else:
            return num_pixel

    @classmethod
    def _check_shape(x, y, d):
        if np.shape(x) == np.shape(y) == np.shape(d):
            pass
        else:
            raise Exception(f"x, y, d must have same shape. \n"
                            f"x: {np.shape(x)}\n"
                            f"y: {np.shape(y)}\n"
                            f"d: {np.shape(d)}")


class ImportFromSerial(HexImageBase):
    """
    Make X and Y coordinate notated image data from serialized hexagonal image data. Number of rows (`d_nrow`),
    stack sequence (`d_sequence`), and stack of the serialized data with 1-dimensional shape (`d_stack`) essentially
    required. Other attributes can be left optional.

    Attributes
    ----------
    image_data : np.ndarray | None
        Serialized stack of the data. Shape should be 1-dimensional.
    d_sequence : np.ndarray | None
        Stack sequence. Valid format for `d_sequence` looks like below:
        >>> np.array([
        ...    [262, 0.0],  # Even number row column number is 262. Offset is 0 in row direction.
        ...    [261, 0.5],  # Odd number row column number is 261. Offset is 0.5 in row direction.
        ... ])
    d_nrow : np.ndarray | None
        Number of rows.
    x_step : float
        stepsize for x-direction.
    y_step : float
        stepsize for y-direction.

    """

    def __init__(self, image_data:np.ndarray,
                 d_sequence:np.ndarray | None = None,
                 d_nrow:int | None = None,
                 d_ncol_odd:int | None = None,
                 d_ncol_even:int | None = None,
                 d_ncol_even_offset:float | None = None,
                 d_ncol_odd_offset:float | None = None,
                 x_step=None,
                 y_step=None):
        """
        Parameters
        ----------
        image_data : np.ndarray

        d_sequence : np.ndarray, optional

        d_nrow : int, optional

        x_step : float, optional
            Step size between neighbouring hexagonal patch in x-direction (column-column distance). You can specify this
            value manually. Otherwise infered from y_step or set to be 1 if y_step not available.
        y_step : float, optional
            Step size between neighbouring hexagonal patch in y-direction (row-row distance). You can specify this
            value manually. Otherwise infered from x_step or set to be $\frac{2}{\sqrt(3)}$ if x_step not available.
        """
        super().__init__()

        if (d_sequence is None):
            if None in (d_nrow, d_ncol_odd, d_ncol_even, d_ncol_even_offset, d_ncol_odd_offset):
                raise ValueError("d_nrow, d_ncol_odd, d_ncol_even, d_ncol_even_offset, d_ncol_odd_offset are required "
                                "if d_sequence is not provided.")
            else:
                self.image_columns_offset_odd: float | None = d_ncol_odd_offset
                self.image_columns_offset_even: float | None = d_ncol_even_offset
                self.image_columns_number_odd: int | None = d_ncol_odd
                self.image_columns_number_even: int | None = d_ncol_even
        else:
            if d_nrow is None:
                raise ValueError("d_nrow must be provided if d_sequence is provided.")
            else:
                self.image_columns_number_even: int | None = d_sequence[0][0]
                self.image_columns_number_odd: int | None = d_sequence[1][0]
                self.image_columns_offset_even: float | None = d_sequence[0][1]
                self.image_columns_offset_odd: float | None = d_sequence[1][1]

        # Set parent class attributes.
        self.image_data: np.ndarray | None = image_data
        self.image_rows_number: int | None = d_nrow

        # Evaluate X and Y step value provided.
        self.image_dx: float = x_step
        self.image_dy: float = y_step

        # Data validation and population.
        self._set_xystep()  # Check X, Y data. If not available, try to set by default value.
        self._test_and_correct_error()  # Check `d_sequence` is correct. If not, try to fix.
        self.compile_image()  # Compile image data.
        self._recalculate()  # Recalculate X, Y, D data.


    def compile_image(self) -> NoReturn:
        self.x_coordinates: np.ndarray = np.empty(0)
        self.y_coordinates: np.ndarray = np.empty(0)

        for row_index in range(self.image_rows_number):
            if row_index % 2 == 0:
                row_columns = self.image_columns_number_even
                row_offset = self.image_columns_offset_even
            else:
                row_columns = self.image_columns_number_odd
                row_offset = self.image_columns_offset_odd

            # Y = Rows, X = Columns.
            x_array_temp = np.array([(self.image_dx * x) + row_offset for x in range(row_columns)])  # [0.5, 1.5, 2.5, ...]
            y_array_temp = np.array([self.image_dy * row_index for _ in range(row_columns)])  # [0.5, 1.5, 2.5, ...]

            # Append to the main X & Y coordinate stack.
            self.x_coordinates = np.append(self.x_coordinates, x_array_temp)
            self.y_coordinates = np.append(self.y_coordinates, y_array_temp)

        del x_array_temp
        del y_array_temp

class FromImage(HexImageBase):
    def __init__(self, array_like_2d):
        self.x, self.y = array_like_2d.shape
        pass
