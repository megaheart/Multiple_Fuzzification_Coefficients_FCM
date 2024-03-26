import numpy as np
import math
from typing import List, Tuple, Callable, Union

class M2_PrecalculationTable():
    """
    Provide m2 precalculation table for sSMC-FCM algorithm

    Attributes
    ----------
    _m2_table : np.ndarray
        1D Numpy array of precalculated m2 values
    """

    def __init__(self, m : float, U_expect : float, precision : int = 3, type : Union["int", "float"] = "int", epsilon : float = 1e-4):
        """
        Initialize the precalculated m2 table

        Parameters
        ----------
        m : float
            Fuzziness coefficient of supervised points
        U_expect : float
            Expected value of membership degree
        precision : int
            Precision of m2, number of decimal places. Default is 3
        type : "int" | "float"
            Type of m2, either 'int' or 'float'. Default is 'int'
        epsilon : float
            Threshold of zero distance. Default is 1e-4
        """
        self._m2_table = np.zeros(int(10 ** precision))
        if type == "int":
            self.__init_int_table(m, U_expect)
        elif type == "float":
            self.__init_float_table(m, U_expect, epsilon)
        else:
            raise ValueError("Type must be either 'int' or 'float'")

    def __getitem__(self, key : float) -> float:
        """
        Get the precalculated m2 value

        Parameters
        ----------
        key : double
            The value of m

        Returns
        -------
        double
            The precalculated m2 value
        """
        n = len(self._m2_table)
        index = int(min(key * n, n - 1))
        return self._m2_table[index]
    
    def __init_int_table(self, m:float, U_expect:float):
        n = len(self._m2_table)
        previous_m2 = m

        for i in range(n - 1, -1, -1):
            u = i / n
            self._m2_table[i] = previous_m2 = self.calculate_m2_integer(m, previous_m2, U_expect, u)
    
    def __init_float_table(self, m:float, U_expect:float, epsilon:float):
        n = len(self._m2_table)
        previous_m2 = m

        for i in range(n - 1, -1, -1):
            u = i / n
            self._m2_table[i] = previous_m2 = self.calculate_m2_float(m, previous_m2, U_expect, u, epsilon)

    @staticmethod
    def calculate_m2_integer(m:float, start_m2:float, alpha:float, Uik:float) -> float:
        right = m * ((1 - alpha) / (1 / Uik - 1)) ** (m - 1)
        m1 = max(start_m2, -1 / math.log(alpha))  # Start value of M2
        left = m1 * (alpha ** (m1 - 1))

        while left > right:
            m1 += 1
            left = m1 * (alpha ** (m1 - 1))
        
        return m1

    @staticmethod
    def calculate_m2_float(m:float, start_m2:float, alpha:float, Uik:float, epsilon:float) -> float:
        right = m * ((1 - alpha) / (1 / Uik - 1)) ** (m - 1)
        M2_l = max(start_m2, -1 / math.log(alpha))  # Start value of M2
        left_l = M2_l * (alpha ** (M2_l - 1))
        if left_l <= right:
            return M2_l
        M2_incr = 1
        M2_r = M2_l + M2_incr
        left_r = M2_r * (alpha ** (M2_r - 1))
        while left_r > right:
            M2_incr *= 2
            M2_r = M2_l + M2_incr
            left_r = M2_r * (alpha ** (M2_r - 1))

        while (M2_r - M2_l) > epsilon:
            M2_mid = (M2_l + M2_r) / 2
            left_mid = M2_mid * (alpha ** (M2_mid - 1))
            if left_mid > right:
                M2_l = M2_mid
            else:
                M2_r = M2_mid

        return M2_r