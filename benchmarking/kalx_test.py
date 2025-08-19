from kalxulus import Kalxulus
import numpy as np

x = np.linspace(0, 3, 200)**2 #to make nonuniform

kalx = Kalxulus(
    x_values=x,  # nonuniform-ish
    derivative_order=1,
    num_points=5,
    solver="scipy",
    eo=1e-7,
    num_points_guess=2,
    max_points=21,
    only_nonuniform=False,
    uniform_tol=1e-12,
    coeff_tolerance=1e-8,
    max_iters_factor=3,
)

num_points, error = kalx.compute_stencil_requirements()
print(num_points, error)
assert num_points.shape == error.shape == (kalx.x_values.size,)