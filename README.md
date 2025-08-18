# Kalxulus

[![Lint/Test](https://github.com/Kalixo-Labs/kalxulus/actions/workflows/lint-and-test.yml/badge.svg?branch=main)](https://github.com/Kalixo-Labs/kalxulus/actions/workflows/lint-and-test.yml)
[![Docs](https://github.com/Kalixo-Labs/kalxulus/actions/workflows/generate-docs.yml/badge.svg?branch=main)](https://github.com/Kalixo-Labs/kalxulus/actions/workflows/generate-docs.yml)

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

Kalxulus is a high-precision numerical toolset for computing derivatives and integrals of discrete datasets. It
calculates N-th order derivatives and integrals using an arbitrary number of points via finite difference methods. By
leveraging point-based calculations, it delivers extremely accurate results for scientific and engineering applications
requiring precise derivatives and integrals from x-y data.

## Features

- High-precision numerical differentiation of discrete data
- Accurate integration methods for numerical datasets
- Support for both uniform and non-uniform grid spacing
- Built on robust numerical libraries (NumPy and SciPy)
- Optimized for scientific and engineering applications

## Installation

```
pip install kalixo-kalxulus
```

## Quick Start

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

