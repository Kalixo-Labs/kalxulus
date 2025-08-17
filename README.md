# Kalxulus

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-GPLv3-green)

A toolset for the calculation of high-precision derivatives and integrals of numerical data sets.

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

