"""
Kalxulus - High-precision derivatives and integrals for discrete datasets.

This package provides utilities to compute numerical derivatives and integrals
on 1-D datasets using finite-difference schemes for arbitrary, potentially
non-uniform grids. It supports both dense (NumPy) and sparse (SciPy) backends.

Quick start:
    ```
    from kalxulus import Kalxulus, derivative, integral

    # Class-based API
    kx = Kalxulus(x_values=x, derivative_order=1, num_points=8, solver="scipy")
    dy = kx.derivative(y)

    # Functional convenience API
    dy = derivative(x, y, derivative_order=1, num_points=8, solver="scipy")
    y_int = integral(x, y, integration_order=1, num_points=8, solver="scipy", constant=0.0)
    ```
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional

# Prefer version injected by VCS (e.g., hatch-vcs -> src/kalxulus/_version.py)
try:
    from ._version import __version__  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    # Fallback to installed package metadata
    try:
        from importlib.metadata import version, PackageNotFoundError  # Python 3.8+
    except Exception:  # pragma: no cover
        # Ultimate fallback if importlib.metadata unavailable
        __version__ = "0.0.0"  # type: ignore[assignment]
    else:  # pragma: no cover
        try:
            __version__ = version("kalixo-kalxulus")  # must match [project].name in pyproject.toml
        except PackageNotFoundError:
            __version__ = "0.0.0"  # type: ignore[assignment]

from .kalxulus import Kalxulus  # public class
from . import kalxulus as kalxulus  # re-export submodule for advanced users


def derivative(
    x_values: Sequence[float],
    y_values: Sequence[float],
    derivative_order: int = 1,
    num_points: int = 8,
    solver: str = "scipy",
    tolerance: float = 1e-8,
):
    """Compute the numerical derivative of y(x) using a finite-difference scheme.

    This is a convenience wrapper around Kalxulus for one-shot differentiation.

    Args:
        x_values: 1-D sequence of x-coordinates (non-empty). Can be non-uniform.
        y_values: 1-D sequence of y-values aligned with x_values (same length).
        derivative_order: Non-negative integer derivative order (default: 1).
        num_points: Number of stencil points (>= 1). The effective stencil length is num_points + 1.
        solver: Backend to use, one of {"numpy", "scipy"} (case-insensitive).
        tolerance: Positive threshold to zero-out small coefficients during generation.

    Returns:
        A NumPy array containing the derivative values with the same length as x_values.

    Raises:
        ValueError: If inputs are empty or lengths do not match.
        TypeError: If argument types are invalid.
    """
    import numpy as np

    if x_values is None or y_values is None:
        raise ValueError("x_values and y_values must be provided.")
    if len(x_values) == 0:
        raise ValueError("x_values must be non-empty.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    kx = Kalxulus(
        x_values=np.asarray(x_values),
        derivative_order=derivative_order,
        num_points=num_points,
        solver=solver,
        tolerance=tolerance,
    )
    return kx.derivative(np.asarray(y_values))


def integral(
    x_values: Sequence[float],
    y_values: Sequence[float],
    integration_order: int = 1,
    num_points: int = 8,
    solver: str = "scipy",
    tolerance: float = 1e-8,
    constant: float = 0.0,
):
    """Compute the numerical integral (antiderivative) of y(x).

    This convenience wrapper uses the pseudoinverse of the derivative operator
    built by Kalxulus to perform one or more integrations.

    Args:
        x_values: 1-D sequence of x-coordinates (non-empty). Can be non-uniform.
        y_values: 1-D sequence of y-values aligned with x_values (same length).
        integration_order: Number of times to integrate (>= 0). Default: 1.
        num_points: Number of stencil points (>= 1). Effective stencil length is num_points + 1.
        solver: Backend to use, one of {"numpy", "scipy"} (case-insensitive).
        tolerance: Positive threshold to zero-out small coefficients during generation.
        constant: Integration constant added after shifting the first value to zero.

    Returns:
        A NumPy array containing the integrated values with the same length as x_values.

    Raises:
        ValueError: If inputs are empty or lengths do not match.
        TypeError: If argument types are invalid.
    """
    import numpy as np

    if x_values is None or y_values is None:
        raise ValueError("x_values and y_values must be provided.")
    if len(x_values) == 0:
        raise ValueError("x_values must be non-empty.")
    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length.")

    kx = Kalxulus(
        x_values=np.asarray(x_values),
        derivative_order=max(1, int(integration_order)),  # internal order used to construct operator
        num_points=num_points,
        solver=solver,
        tolerance=tolerance,
    )
    return kx.integral(np.asarray(y_values), integration_order=int(integration_order), num_points=num_points, constant=constant)


__all__ = [
    "Kalxulus",
    "kalxulus",
    "derivative",
    "integral",
    "__version__",
]
