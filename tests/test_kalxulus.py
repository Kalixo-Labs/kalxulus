import numpy as np
import pytest

from kalxulus import Kalxulus


class TestKalxulus:
    @pytest.fixture
    def sample_data(self):
        x = np.linspace(0, np.pi, 100)
        y = np.sin(x)
        return x, y

    def test_initialization(self):
        x = np.linspace(0, 10, 100)
        kalx = Kalxulus(x_values=x, derivative_order=1, num_points=13)
        assert isinstance(kalx, Kalxulus)
        assert kalx.derivative_order == 1
        assert kalx.num_points == 13

    def test_invalid_initialization(self):
        x = np.linspace(0, 10, 100)
        with pytest.raises(ValueError):
            Kalxulus(x_values=x, derivative_order=-1, num_points=13)
        with pytest.raises(ValueError):
            Kalxulus(x_values=x, derivative_order=1, num_points=0)
        with pytest.raises(ValueError):
            Kalxulus(x_values=x, derivative_order=-1, num_points=10)

    def test_derivative_calculation(self, sample_data):
        x, y = sample_data
        kalx = Kalxulus(x_values=x, derivative_order=1, num_points=13)
        dy_dx = kalx.derivative(y)
        assert dy_dx.shape == y.shape
        np.testing.assert_allclose(dy_dx, np.cos(x), atol=1e-4)


    def test_second_derivative(self, sample_data):
        x, y = sample_data
        kalx = Kalxulus(x_values=x, derivative_order=2, num_points=13)
        d2y_dx2 = kalx.derivative(y)
        assert d2y_dx2.shape == y.shape
        np.testing.assert_allclose(d2y_dx2, -np.sin(x), atol=1e-4)

    def test_integral_calculation(self, sample_data):
        x, y = sample_data
        kalx = Kalxulus(x_values=x, derivative_order=1, num_points=13)
        y_int = kalx.integral(y, constant=0.0)
        assert y_int.shape == y.shape
        np.testing.assert_allclose(y_int, -np.cos(x) + 1, atol=1e-4)

    def test_invalid_input_derivative(self):
        x = np.linspace(0, 10, 100)
        kalx = Kalxulus(x_values=x, derivative_order=1, num_points=13)
        with pytest.raises(ValueError):
            kalx.derivative(np.array([1, 2, 3]))  # Wrong input size

    def test_invalid_input_integral(self):
        x = np.linspace(0, 10, 100)
        kalx = Kalxulus(x_values=x, derivative_order=1, num_points=13)
        with pytest.raises(ValueError):
            kalx.integral(np.array([1, 2, 3]))  # Wrong input size
