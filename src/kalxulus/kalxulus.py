"""Kalxulus: high-precision numerical derivatives and integrals for discrete datasets.

This module provides the Kalxulus class, which computes finite-difference
derivative and integral operators on nonuniform grids, with support for
dense (NumPy) and sparse (SciPy) solver backends.

Basic Usage:
    ```
    import numpy as np
    from kalxulus import Kalxulus

    # Generate sample data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Initialize Kalxulus instance
    kalx = Kalxulus(x_values=x, derivative_order=1, num_points=8)

    # Compute first derivative
    dy_dx = kalx.derivative(y)

    # Compute integral (antiderivative)
    y_int = kalx.integral(y, constant=0.0)
    ```

"""

from __future__ import annotations

import argparse
import itertools
import sys
from math import factorial
from typing import Literal, Optional, Sequence, Union

import numpy as np
from scipy.sparse import csr_matrix


class Kalxulus:
    def __init__(
        self,
        x_values: Optional[Sequence[float] | np.ndarray] = None,
        derivative_order: int = 1,
        num_points: int = 8,
        solver: Literal["numpy", "scipy"] = "scipy",
        tolerance: float = 1e-8,
    ) -> None:
        """Compute numerical derivatives and integrals over 1-D sample points.

        Builds finite-difference derivative matrices for an arbitrary 1-D grid
        (x_values), using either a dense NumPy backend or a sparse SciPy backend.
        Also supports numerical integration via pseudoinverse of the derivative operator.

        Args:
            x_values: 1-D numeric sequence (list, tuple, or NumPy array) of x-coordinates.
                Values are converted to a float np.ndarray. Must be non-empty.
            derivative_order: Non-negative integer derivative order to build by default.
            num_points: Number of stencil points used in the local finite-difference
                scheme (must be >= 1). The effective stencil length is num_points + 1.
            solver: Backend used to store/apply the operator. Either "numpy" (dense)
                or "scipy" (sparse), case-insensitive.
            tolerance: Positive threshold used to zero-out small coefficients
                during coefficient generation.

        Attributes:
            x_values (np.ndarray): Sorted as provided, stored as 1-D float array.
            derivative_order (int): Default derivative order used when none is specified.
            num_points (int): Default number of stencil points.
            solver (str): Normalized solver backend, one of {"numpy", "scipy"}.
            tolerance (float): Coefficient zeroing threshold.
            G (dict[tuple[int, int], np.ndarray]): Cache for internal combinatorial masks.
            derivative_coefficients (dict[tuple[int, int], np.ndarray | scipy.sparse.csc_matrix]):
                Cached derivative operators keyed by (derivative_order, num_points).
            integration_coefficients (dict[tuple[int, int], np.ndarray]):
                Cached pseudoinverse operators keyed by (integration_order, num_points).

        Raises:
            TypeError: If argument types are invalid.
            ValueError: If argument values are out of allowed ranges, or x_values is empty.
        """
        # x_values
        if x_values is None:
            self.x_values = np.array([], dtype=float)
        elif isinstance(x_values, np.ndarray):
            if x_values.ndim != 1:
                raise ValueError("x_values must be a 1-D array.")
            self.x_values = x_values.astype(float, copy=False)
        elif isinstance(x_values, Sequence) and not isinstance(x_values, (str, bytes)):
            try:
                arr = np.asarray(x_values, dtype=float)
            except (TypeError, ValueError) as e:
                raise TypeError("x_values must be a sequence of numbers.") from e
            if arr.ndim != 1:
                raise ValueError("x_values must be a 1-D sequence.")
            self.x_values = arr
        else:
            raise TypeError(
                "x_values must be a 1-D sequence (list/tuple) or numpy.ndarray."
            )

        # derivative_order
        if not isinstance(derivative_order, int):
            raise TypeError("derivative_order must be an integer.")
        derivative_order = int(derivative_order)
        if derivative_order < 0:
            raise ValueError("derivative_order must be >= 0.")
        self.derivative_order = derivative_order

        # num_points
        if not isinstance(num_points, int):
            raise TypeError("num_points must be an integer.")
        num_points = int(num_points)
        if num_points < 1:
            raise ValueError("num_points must be >= 1.")
        self.num_points = num_points

        # solver
        if not isinstance(solver, str):
            raise TypeError("solver must be a string: 'numpy' or 'scipy'.")
        solver_norm = solver.strip().lower()
        if solver_norm not in {"numpy", "scipy"}:
            raise ValueError("solver must be either 'numpy' or 'scipy'.")
        self.solver = solver_norm

        # tolerance
        if not isinstance(tolerance, float):
            raise TypeError("tolerance must be a real number.")
        tolerance = float(tolerance)
        if tolerance <= 0:
            raise ValueError("tolerance must be > 0.")
        self.tolerance = tolerance

        self.G = {}
        self.derivative_coefficients = {}
        self.integration_coefficients = {}
        self.solver = str(solver)
        if len(x_values) > 0:
            self.x_values = np.asarray(x_values)
            if num_points >= len(self.x_values):
                num_points = len(self.x_values) - 1
        else:
            raise ValueError("The length of x_values must be greater than zero.")
        self.derivative_order = int(derivative_order)
        self.num_points = int(num_points)
        self.gen_coefficients(int(derivative_order), int(num_points))

    def __call__(self):
        # """Return the internal state as a dictionary.
        #
        # Returns:
        #     dict: The instance __dict__ containing configuration and caches.
        # """
        return self.__dict__

    def __G_function(self, ld, derivative_order):
        # Private helper; no public docstring required.
        if (derivative_order, ld) in self.G:
            return self.G[(derivative_order, ld)]
        else:
            self.G[(derivative_order, ld)] = np.array(
                list(itertools.combinations(range(ld), ld - (derivative_order - 1))),
                dtype="int",
            )
            return self.G[(derivative_order, ld)]

    def __a_function(self, delta, derivative_order):
        # Private helper; no public docstring required.
        return np.prod(
            np.array(delta)[self.__G_function(len(delta), derivative_order)], axis=1
        ).sum()

    def gen_coefficients(self, derivative_order: int, num_points: int):
        # """Generate and cache derivative operator coefficients.
        #
        # Builds the finite-difference matrix for a given derivative order and
        # stencil size over the provided x_values grid. The resulting operator is
        # stored in derivative_coefficients[(derivative_order, num_points)] as either
        # a dense ndarray ("numpy") or a CSC sparse matrix ("scipy").
        #
        # Args:
        #     derivative_order (int): Non-negative derivative order.
        #     num_points (int): Number of stencil points (>= 1). Effective
        #         stencil length is num_points + 1.
        #
        # Returns:
        #     None
        #
        # Raises:
        #     TypeError: If parameters are not integers or are out of range.
        # """
        if (type(derivative_order) not in [int]) or (derivative_order < 0):
            raise TypeError("derivative_order must be a positive integer value.")
        if (type(num_points) not in [int]) or (num_points < 1):
            raise TypeError("num_points must be a positive integer value >= 1.")

        if derivative_order == 0:
            if self.solver == "numpy":
                self.derivative_coefficients[(derivative_order, num_points)] = np.eye(
                    len(self.x_values)
                )
                return None
            elif self.solver == "scipy":
                self.derivative_coefficients[(derivative_order, num_points)] = (
                    csr_matrix(np.eye(len(self.x_values))).tocsc()
                )
                return None

        N = len(self.x_values) - 1
        lx = num_points + 1
        K = num_points // 2
        coeffs = np.zeros((N + 1, lx), dtype="float64")
        derivative_order_factorial = (-1) ** derivative_order * factorial(
            derivative_order
        )
        factors = np.minimum(np.maximum(np.arange(N + 1) - K, 0), N - num_points)

        ind = np.array(np.outer(factors, np.ones(lx)) + np.arange(lx), dtype=int)
        jk = np.indices((lx, lx))
        dmaski = np.eye(lx, dtype=bool)
        for i in range(N + 1):
            prod = np.array(
                (self.x_values[jk[1] + factors[i]] - self.x_values[jk[0] + factors[i]])
            )
            np.fill_diagonal(prod, 1)
            prods = prod.prod(axis=1)
            dmask = np.copy(dmaski)
            dmask[:, i - factors[i]] = True
            dmask[i - factors[i], :] = True
            delta = np.ma.array(
                self.x_values[jk[1] + factors[i]] - self.x_values[i], mask=dmask
            )
            for j in range(lx):
                if j == i - factors[i]:
                    pass
                else:
                    coeffs[i, j] = (
                        self.__a_function(delta[j].compressed(), derivative_order)
                        / prods[j]
                    )
            coeffs[i, np.abs(coeffs[i, :]) < self.tolerance] = 0.0
            coeffs[i, i - factors[i]] = -1.0 * coeffs[i].sum()
        coeffs *= derivative_order_factorial
        if self.solver == "numpy":
            mat = np.zeros([len(coeffs)] * 2)
            for i in range(len(coeffs)):
                for index in range(len(ind[i])):
                    mat[i, ind[i, index]] = coeffs[i, index]
            self.derivative_coefficients[(derivative_order, num_points)] = mat.copy()
        if self.solver == "scipy":
            indptr = np.array(
                [0] + np.cumsum(np.array([len(i) for i in coeffs])).tolist()
            )
            mat = csr_matrix(
                (coeffs.ravel(), ind.ravel(), indptr), shape=(N + 1, N + 1)
            ).tocsc()
            self.derivative_coefficients[(derivative_order, num_points)] = mat.copy()

        return None

    def derivative(
        self,
        y_values: Union[Sequence[float] | np.ndarray],
        derivative_order: int = None,
        num_points: int = None,
    ) -> np.ndarray:
        """Apply the derivative operator to y-values.

        Computes the derivative of y_values sampled at x_values using the
        cached operator for the requested derivative_order and num_points.
        If not already cached, the coefficients are generated on demand.

        Args:
            y_values (list | np.ndarray): 1-D array-like of function values aligned with x_values.
            derivative_order (int, optional): Derivative order to apply. Defaults to self.derivative_order.
            num_points (int, optional): Stencil size to use. Defaults to self.num_points.

        Returns:
            np.ndarray: The differentiated values with the same length as x_values.

        Raises:
            TypeError: If derivative_order or num_points are not valid integers.
            ValueError: If orders are out of allowed ranges.
        """
        if derivative_order is None:
            derivative_order = self.derivative_order
        if num_points is None:
            num_points = self.num_points

        if (type(derivative_order) not in [int]) or (derivative_order < 0):
            raise TypeError("derivative_order must be a positive integer value.")
        if (type(num_points) not in [int]) or (num_points < 1):
            raise TypeError("num_points must be a positive integer value >= 1.")

        if (derivative_order, num_points) not in self.derivative_coefficients:
            self.gen_coefficients(int(derivative_order), int(num_points))
        if self.solver == "numpy":
            return np.inner(
                self.derivative_coefficients[(derivative_order, num_points)],
                np.asarray(y_values),
            )
        if self.solver == "scipy":
            return self.derivative_coefficients[(derivative_order, num_points)].dot(
                np.asarray(y_values)
            )

    def __integration_function(self, inverted_matrix, yp, constant):
        # Private helper; no public docstring required.
        y_inv = np.inner(inverted_matrix, np.asarray(yp))
        return y_inv - y_inv[0] + constant

    def integral(
        self, y_values, integration_order=None, num_points=None, constant=0.0
    ) -> np.ndarray:
        """Numerically integrate y-values by inverting the derivative operator.

        Uses the pseudoinverse of the derivative operator (for the given
        integration_order and num_points) to obtain an antiderivative. If the
        operator is not cached, it is generated (or derived from an existing
        derivative operator) and cached.

        Args:
            y_values (array-like): 1-D sequence of function values aligned with x_values.
            integration_order (int, optional): Number of times to integrate. Defaults to self.derivative_order.
            num_points (int, optional): Stencil size used for the operator. Defaults to self.num_points.
            constant (float, optional): Integration constant added after shifting the first value to zero. Defaults to 0.0.

        Returns:
            np.ndarray: The integrated values after applying the specified number of integrations.

        Raises:
            TypeError: If integration_order or num_points are not valid integers.
            ValueError: If orders are out of allowed ranges.
        """
        if integration_order is None:
            integration_order = self.derivative_order
        if num_points is None:
            num_points = int(self.num_points)

        if (type(integration_order) not in [int]) or (integration_order < 0):
            raise TypeError("integration_order must be a positive integer value.")
        if (type(num_points) not in [int]) or (num_points < 1):
            raise TypeError("num_points must be a positive integer value >= 1.")

        if (integration_order, num_points) not in self.integration_coefficients:
            if (integration_order, num_points) not in self.derivative_coefficients:
                self.gen_coefficients(integration_order, num_points)
            if self.solver == "numpy":
                self.integration_coefficients[(integration_order, num_points)] = (
                    np.linalg.pinv(
                        self.derivative_coefficients[(integration_order, num_points)]
                    )
                )  # 1
            if self.solver == "scipy":
                self.integration_coefficients[(integration_order, num_points)] = (
                    np.linalg.pinv(
                        self.derivative_coefficients[
                            (integration_order, num_points)
                        ].toarray()
                    )
                )  # 2
        integrated_values = np.asarray(y_values).copy()

        for ints in range(integration_order):
            integrated_values = self.__integration_function(
                inverted_matrix=self.integration_coefficients[
                    (integration_order, num_points)
                ],
                yp=integrated_values,
                constant=constant,
            )
        return integrated_values


# ==================================================================#
# ==================TEST CODE=======================================#
# ==================================================================#
if __name__ == "__main__":
    description = "Kalixo Derivative and Integral Toolkit"
    ap = argparse.ArgumentParser(
        description=description, epilog="All Rights Reserved, Kalixo 2018"
    )

    ap.add_argument(
        "-i",
        "--infile",
        type=argparse.FileType("r"),
        dest="INFILE",
        default=sys.stdin,
        help="Name of input data file. Must be in delimted, two column format. You may also pipe data directly to the script in the same format, ommiting the '-i' command. (Default: stdin)",
    )

    ap.add_argument(
        "-o",
        "--outfile",
        dest="OUTFILE",
        type=argparse.FileType("w"),
        default=sys.stdout,
        help="Name of output data file. Can also be piped to stdout by ommiting the '-o' command. (Default: stdout)",
    )

    ap.add_argument(
        "-d",
        "--deliminter",
        dest="DELIM",
        type=str,
        default="  ",
        help="Delimiter to use in file output. (Default: two spaces)",
    )

    ap.add_argument(
        "-DO",
        "--order",
        dest="DO",
        type=int,
        default=1,
        help="Derivative order of the calculation (Default: 1)",
    )

    ap.add_argument(
        "-NPT",
        "--numpoints",
        dest="NPT",
        type=int,
        default=5,
        help="Number of points to use in the derivative calculation (Default: 5)",
    )

    ap.add_argument(
        "-m",
        "--method",
        dest="METHOD",
        type=str,
        default="derivative",
        help="Whether to derivate or integrate the data. \n\nAvailable methods are: ['derivative', 'derivate', 'der', 'd', 'integral', 'integrate', 'int','i']. (Default 'derivative')",
    )

    ap.add_argument(
        "-s",
        "--solver",
        dest="SOLVER",
        type=str,
        default="numpy",
        help="The solver type to use for inverting the derivative matrix.  Available methods are: ['numpy','scipy']. (Default: numpy)",
    )

    # ap.add_argument("-p", "--plot", dest='PLOT', type=bool, default=False, \
    #                 help="Whether to output a plot of the derivation/integration [True,False]. (Default: False)")

    args = vars(ap.parse_args())

    infile = args["INFILE"]
    outfile = args["OUTFILE"]
    DELIM = args["DELIM"]
    DO = args["DO"]
    NPT = args["NPT"]
    METHOD = args["METHOD"]
    SOLVER = args["SOLVER"]

    x, y = np.loadtxt(infile, unpack=True)
    print(f"Sucessfully loaded {infile}")
    x_length = int(x.shape[0])

    # 	import numpy as np

    kalx = Kalxulus(x_values=x, num_points=NPT, derivative_order=DO, solver=SOLVER)
    if METHOD in ["derivative", "der", "derivate", "d"]:
        outdata = kalx.derivative(y)

    elif METHOD in ["integral", "int", "integrate", "i"]:
        Y = y.copy()
        for i in range(DO):
            Y = kalx.integral(Y, integration_order=1)
        outdata = Y

    # 	file = open(outfile, 'w')

    file = outfile
    for i in range(len(outdata)):
        file.write("%0.5f" % x[i] + DELIM + "%0.5f\n" % outdata[i])
    # 	file.close()
    print(f"Wrote file output to {outfile}")

# ==================================================================#
# ==================================================================#
# ==================================================================#
