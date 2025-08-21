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

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class StencilConfig:
    eo: float = 1e-6  # target error threshold
    num_points_guess: int = 5  # initial stencil size in points
    max_points: int = 11  # maximum stencil size in points
    only_nonuniform: bool = True
    uniform_tol: float = 1e-12
    max_iters_factor: int = 2  # safety cap: max_iters = factor * max_points


class ErrorEstimatorMixin:
    """
    Expects: self.x_values: np.ndarray
    Public API returns and accepts num_points (stencil size in POINTS).
    """

    def __init__(self, stencil_config: Optional[StencilConfig] = None):
        # Note: we *don’t* call super().__init__ here to avoid interfering with your
        # existing Kalxulus.__init__. Kalxulus will call this explicitly.
        self._stencil_config = stencil_config or StencilConfig()

    # --- Public API ---

    def set_stencil_config(self, **kwargs) -> "ErrorEstimatorMixin":
        for k, v in kwargs.items():
            if not hasattr(self._stencil_config, k):
                raise AttributeError(f"Unknown stencil config option: {k}")
            setattr(self._stencil_config, k, v)
        return self

    def compute_stencil_requirements(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        num_points : (n,) int array   # stencil size in points
        error      : (n,) float array
        """
        x = np.asarray(self.x_values, dtype=float)
        cfg = self._stencil_config
        n = x.size

        num_points = np.full(n, max(2, int(cfg.num_points_guess)), dtype=int)
        num_points = np.minimum(num_points, max(2, int(cfg.max_points)))
        errs = np.full(n, np.nan, dtype=float)

        to_process = np.ones(n, dtype=bool)
        if cfg.only_nonuniform:
            to_process = self._nonuniform_points(
                x, window_points=int(cfg.num_points_guess), tol=cfg.uniform_tol
            )

        idxs0 = np.where(to_process)[0]
        if idxs0.size > 0:
            # initial guess is uniform, so take any representative entry
            n_intervals0 = int(num_points[idxs0[0]]) - 1
            E0, ok0 = self._error_for_batch(x, idxs0, n_intervals0)
            errs[idxs0] = E0

        done = ~to_process | (errs <= cfg.eo) | (num_points >= cfg.max_points)

        max_iters = cfg.max_iters_factor * cfg.max_points
        iters = 0
        while True:
            iters += 1
            if iters > max_iters:
                break
            need = ~done
            if not np.any(need):
                break
            if np.all(num_points[need] >= cfg.max_points):
                break

            num_points_next = np.minimum(num_points + 1, cfg.max_points)

            for s in np.unique(num_points_next[need]):
                grp = need & (num_points_next == s)
                if not np.any(grp):
                    continue
                idxs = np.where(grp)[0]
                n_intervals = int(s) - 1
                E, ok = self._error_for_batch(x, idxs, n_intervals)

                replace = np.isnan(errs[idxs]) | (E < errs[idxs]) | (errs[idxs] > cfg.eo)
                replace &= ok
                errs[idxs[replace]] = E[replace]
                num_points[idxs[replace]] = int(s)

            done = ~to_process | (errs <= cfg.eo) | (num_points >= cfg.max_points)

        return num_points, errs

    def compute_stencil_for_index(self, i: int) -> Tuple[int, float]:
        """
        Returns (num_points, error) for index i.
        """
        x = np.asarray(self.x_values, dtype=float)
        cfg = self._stencil_config

        Error = cfg.eo + 1.0
        num_points = min(max(2, int(cfg.num_points_guess)), int(cfg.max_points))

        while Error > cfg.eo and num_points < cfg.max_points:
            num_points += 1
            n_intervals = num_points - 1
            E, ok = self._error_for_batch(x, np.array([i]), n_intervals)
            Error = E[0] if ok[0] else np.nan
            if not np.isfinite(Error):
                break

        return num_points, Error

    # --- Internals ---

    @staticmethod
    def _factor_for_indices(indxs: np.ndarray, n_intervals: int, N: int):
        K = n_intervals // 2
        factor = np.empty_like(indxs)
        left = indxs < K
        mid = (~left) & (indxs <= (N - K - 1))
        right = ~(left | mid)
        factor[left] = 0
        factor[mid] = indxs[mid] - K
        factor[right] = N - n_intervals
        return factor, K

    @staticmethod
    def _nonuniform_points(x: np.ndarray, window_points: int, tol: float) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        n = x.size
        if n < 3 or window_points <= 2:
            return np.ones(n, dtype=bool)
        gaps = np.diff(x)
        win_g = window_points - 1
        if win_g <= 0 or (n - 1 - win_g + 1) <= 0:
            return np.ones(n, dtype=bool)

        gW = np.lib.stride_tricks.sliding_window_view(gaps, win_g)
        gmax = gW.max(axis=1);
        gmin = gW.min(axis=1);
        gmean = gW.mean(axis=1)
        uniform = (gmax - gmin) <= (np.abs(gmean) + 1.0) * tol

        center = window_points // 2
        centers = np.arange(uniform.size) + center

        mask = np.ones(n, dtype=bool)
        mask[:] = True
        mask[centers[uniform]] = False
        mask[:center] = True
        mask[n - center:] = True
        return mask

    @staticmethod
    def _error_for_batch(x: np.ndarray, indxs: np.ndarray, n_intervals: int):
        """
        n_intervals = num_points - 1
        Returns (Error, ok), guaranteed — no warnings on divide-by-zero.
        """
        from math import factorial  # local import: avoids touching module-level imports

        x = np.asarray(x, dtype=float)
        N = len(x) - 1
        M = indxs.size
        lx = n_intervals + 1

        # Preallocate outputs so we always return (Error, ok)
        Error = np.full(M, np.nan, dtype=float)
        ok = np.zeros(M, dtype=bool)

        # Trivial / empty batch
        if lx < 2 or M == 0 or N < 1:
            # Error stays nan for empty; mark ok=True only for structurally valid rows
            if M > 0 and lx >= 2:
                ok[:] = True
                Error[:] = 0.0
            return Error, ok

        # Window placement (left/center/right stencil anchor)
        factor, _K = ErrorEstimatorMixin._factor_for_indices(indxs, n_intervals, N)
        start = factor
        stop = factor + lx
        valid = (start >= 0) & (stop <= len(x))

        # If no valid windows at all, return (Error=nan, ok=False) — already set.
        if not np.any(valid):
            return Error, ok

        # Work only on valid rows
        v_idx = np.where(valid)[0]
        v_start = start[v_idx]
        v_indxs = indxs[v_idx]

        cols = np.arange(lx)
        W = v_start[:, None] + cols[None, :]  # (V, lx) indices
        xw = x[W]  # (V, lx)
        idx0 = v_indxs - v_start  # (V,)

        # fac1 = 1/(n_intervals+1)!
        fac1 = 1.0 / factorial(lx)

        # dt = xw - x[i]
        xi = x[v_indxs]
        dt = xw - xi[:, None]  # (V, lx)

        # fac2: product of dt excluding the center (achieve by setting center dt=1 first)
        dt2 = dt.copy()
        dt2[np.arange(v_idx.size), idx0] = 1.0
        fac2 = np.prod(dt2, axis=1)  # (V,)

        # Pairwise differences D[j,k] = x_j - x_k
        D = xw[:, :, None] - xw[:, None, :]  # (V, lx, lx)
        jj = np.arange(lx)

        # product over k != j: set diag to 1 then product last axis
        D_no_diag = D.copy()
        D_no_diag[:, jj, jj] = 1.0
        row_prod_excl_diag = np.prod(D_no_diag, axis=2)  # (V, lx)

        # denom[j] = (∏_{k≠j} (x_j - x_k)) / (x_j - x_idx0)
        D_j_idx0 = D[np.arange(v_idx.size)[:, None], jj[None, :], idx0[:, None]]  # (V, lx)
        # Safe divide: where denominator is zero, put nan (we'll mask later)
        denom = np.empty_like(row_prod_excl_diag)
        np.divide(row_prod_excl_diag, D_j_idx0, out=denom, where=(D_j_idx0 != 0))

        # terms: zero at j==idx0, else dt[j]**(n_intervals-1) / denom[j]
        term = np.zeros_like(dt)
        mask = (jj[None, :] != idx0[:, None])
        numer = np.zeros_like(dt)
        # When n_intervals == 0, power 0 => 1 on mask; but lx>=2 ⇒ n_intervals>=1 in practice
        if n_intervals >= 1:
            numer[mask] = dt[mask] ** (n_intervals - 1)
        else:
            numer[mask] = 1.0

        # Only fill where denom is finite & nonzero
        safe = mask & np.isfinite(denom) & (denom != 0)
        term[safe] = numer[safe] / denom[safe]

        fac3 = np.sum(term, axis=1)  # (V,)

        E_v = np.abs(fac1 * fac2 * fac3)

        # Write back into full arrays
        Error[v_idx] = E_v
        ok[v_idx] = np.isfinite(E_v)

        # For invalid rows, Error stays nan, ok stays False
        return Error, ok


class Kalxulus(ErrorEstimatorMixin):
    def __init__(
            self,
            x_values: Optional[Sequence[float] | np.ndarray] = None,
            derivative_order: int = 1,
            num_points: int = 8,
            eo: float = 1e-7,
            solver: Literal["numpy", "scipy"] = "scipy",
            coeff_tolerance: float = 1e-8,
            # ---- NEW knobs for the error estimator / stencil search ----
            num_points_guess: Optional[int] = None,  # default to self.num_points if None
            max_points: int = 21,
            only_nonuniform: bool = False,
            uniform_tol: float = 1e-12,
            max_iters_factor: int = 2,
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
            coeff_tolerance: Positive threshold used to zero-out small coefficients
                during coefficient generation.

        Attributes:
            x_values (np.ndarray): Sorted as provided, stored as 1-D float array.
            derivative_order (int): Default derivative order used when none is specified.
            num_points (int): Default number of stencil points.
            solver (str): Normalized solver backend, one of {"numpy", "scipy"}.
            coeff_tolerance (float): Coefficient zeroing threshold.
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
        if not isinstance(coeff_tolerance, float):
            raise TypeError("tolerance must be a real number.")
        coeff_tolerance = float(coeff_tolerance)
        if coeff_tolerance <= 0:
            raise ValueError("tolerance must be > 0.")
        self.coeff_tolerance = coeff_tolerance

        # ---- Stencil / error estimator configuration ----
        # Store eo and max_points at the instance level for convenient access elsewhere:
        if not isinstance(eo, (int, float)):
            raise TypeError("eo must be a real number.")
        self.eo = float(eo)
        if self.eo <= 0:
            raise ValueError("eo must be > 0.")

        if not isinstance(max_points, int):
            raise TypeError("max_points must be an integer.")
        self.max_points = int(max_points)
        if self.max_points < 2:
            raise ValueError("max_points must be >= 2.")

        # Default num_points_guess to current self.num_points if not supplied:
        if num_points_guess is None:
            num_points_guess = int(self.num_points)
        else:
            num_points_guess = int(num_points_guess)
        if num_points_guess < 2:
            raise ValueError("num_points_guess must be >= 2.")

        if not isinstance(only_nonuniform, bool):
            raise TypeError("only_nonuniform must be a boolean.")

        if not isinstance(uniform_tol, (int, float)):
            raise TypeError("uniform_tol must be a real number.")
        uniform_tol = float(uniform_tol)
        if uniform_tol < 0:
            raise ValueError("uniform_tol must be >= 0.")

        if not isinstance(max_iters_factor, int):
            raise TypeError("max_iters_factor must be an integer.")
        max_iters_factor = int(max_iters_factor)
        if max_iters_factor < 1:
            raise ValueError("max_iters_factor must be >= 1.")

        # Build the config for the mixin (all values come from Kalxulus.__init__)
        _cfg = StencilConfig(
            eo=self.eo,
            num_points_guess=num_points_guess,
            max_points=self.max_points,
            only_nonuniform=only_nonuniform,
            uniform_tol=uniform_tol,
            max_iters_factor=max_iters_factor,
        )

        # Explicitly initialize the mixin (keeps your current __init__ logic intact)
        ErrorEstimatorMixin.__init__(self, stencil_config=_cfg)

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
            coeffs[i, np.abs(coeffs[i, :]) < self.coeff_tolerance] = 0.0
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
