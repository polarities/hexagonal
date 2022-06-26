from typing import NoReturn
import numpy as np
import warnings

class ImportFromSerial():
    """
    Make X and Y coordinate notated image data from serialized hexagonal image data. Number of rows (`d_nrow`),
    stack sequence (`d_sequence`), and stack of the serialized data with 1-dimensional shape (`d_stack`) essentially
    required. Other attributes can be left optional.

    Attributes
    ----------
    d_stack : np.ndarray | None
        Serialized stack of the data. Shape should be 1-dimensional.
    d_sequence : np.ndarray | None
        Stack sequence. Valid format for `d_sequence` looks like below:
        >>> np.array([
        ...    [262, 0.0],  # Even number row column number is 262. Offset is 0 in row direction.
        ...    [261, 0.5],  # Even number row column number is 261. Offset is 0.5 in row direction.
        ... ])
    d_nrow : np.ndarray | None
        Number of rows.
    x_step : float
        stepsize for x-direction.
    y_step : float
        stepsize for y-direction.
    """

    def __init__(self, d_stack:np.ndarray, d_sequence:np.ndarray, d_nrow:int, x_step=None, y_step=None):
        """
        Parameters
        ----------
        d_stack : np.ndarray

        d_sequence : np.ndarray

        d_nrow : int

        x_step : float, optional
            Step size between neighbouring hexagonal patch in x-direction (column-column distance). You can specify this
            value manually. Otherwise infered from y_step or set to be 1 if y_step not available.
        y_step : float, optional
            Step size between neighbouring hexagonal patch in y-direction (row-row distance). You can specify this
            value manually. Otherwise infered from x_step or set to be $\frac{2}{\sqrt(3)}$ if x_step not available.
        """
        self.d_stack: np.ndarray | None = d_stack
        self.d_sequence: np.ndarray | None = d_sequence
        self.d_nrow: int | None = d_nrow

        # Evaluate X and Y step value provided.
        self.x_step: float = x_step
        self.y_step: float = y_step

        # Data validation and population.
        self.__set_xystep()  # Check X, Y data. If not available, try to set by default value.
        self.__test_and_correct_error()  # Check `d_sequence` is correct. If not, try to fix.
        self.compile_image()

    @property
    def xydata(self):
        return self.x_stack, self.y_stack, self.d_stack


    def compile_image(self) -> NoReturn:
        self.x_stack: np.ndarray = np.empty(0)
        self.y_stack: np.ndarray = np.empty(0)

        for row_index in range(self.d_nrow):
            sequence = row_index % len(self.d_sequence)  # Making alternating sequence. For example, 0, 1, 0, 1, 0, 1...
            stack_ncol = self.d_sequence[sequence][0]
            stack_offset = self.d_sequence[sequence][1] * self.x_step

            # Y = Rows, X = Columns.
            x_array_temp = self.x_step * (np.array(range(stack_ncol))) + stack_offset  # [0.5, 1.5, 2.5, ...]
            y_array_temp = np.ones(int(stack_ncol)) * row_index * self.y_step  # [0, 0, 0, ...]

            # Append to the main X & Y coordinate stack.
            self.x_stack = np.append(self.x_stack, x_array_temp)
            self.y_stack = np.append(self.y_stack, y_array_temp)

        del x_array_temp
        del y_array_temp

    def __set_xystep(self) -> NoReturn:
        if (self.x_step is None) and (self.y_step is None):
            self.x_step = 1
            self.y_step = (self.x_step * np.sqrt(3)) / 2
        elif (self.x_step is not None) and (self.y_step is None):
            self.y_step = (self.x_step * np.sqrt(3))/2
        elif (self.x_step is None) and (self.y_step is not None):
            self.x_step = self.y_step * (2 / np.sqrt(3))


    def __test_and_correct_error(self):
        if self.__check_hexagonal_dataentry(self.d_sequence[0][0],  # Check EBSD size is right.
                                            self.d_sequence[1][0],
                                            self.d_nrow,
                                            len(self.d_stack),
                                            return_value=False):
            pass
        else:  # If data size is not correct, than try different sequence.
            self.d_sequence = np.flip(self.d_sequence, axis=0)
            if self.__check_hexagonal_dataentry(self.d_sequence[0][0],  # Check EBSD size is right.
                                                self.d_sequence[1][0],
                                                self.d_nrow,
                                                len(self.d_stack),
                                                return_value=False):
                warnings.warn(
                    "Error in provided d_sequence array detected. However it looks can be fixed automatically by "
                    "inverting stacking sequence. This can be happened during parsing .ang file.")
            else:
                raise Exception(f"The number of data is incorrect. \n"
                                f"""- Expected: {self.__check_hexagonal_dataentry(self.d_sequence[0][0],
                                                                                  self.d_sequence[1][0],
                                                                                  self.d_nrow,
                                                                                  len(self.d_stack),
                                                                                  return_value=True)}\n"""
                                f"- Given: {len(self.d_stack)} \n")


    @staticmethod
    def __check_hexagonal_dataentry(ncol_even: int,
                                    ncol_odd: int,
                                    nrow: int,
                                    n_dataentry: int,
                                    return_value: False) -> bool:
        """
        Check whether EBSD data number is correct or not based on the parsed metadata.

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


