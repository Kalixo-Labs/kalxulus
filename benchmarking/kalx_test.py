from src.kalxulus import Kalxulus
import numpy as np

from old_error import errorO

x = np.linspace(0, 2, 20)**2 #to make nonuniform

kalx = Kalxulus(
    x_values=x,  # nonuniform-ish
    derivative_order=1,
    num_points=3,
    solver="scipy",
    eo=1e-7,
    max_points=21,
    only_nonuniform=False,
    uniform_tol=1e-12,
    coeff_tolerance=1e-8,
    max_iters_factor=3,
)
vals = []
errs = []
for i in range(len(x)):
    val, err = errorO(x, indx=i, do=1, eo=1e-7, npt_guess=3)
    vals.append(val)
    errs.append(err)

errs = np.array(errs)

num_points, error = kalx.compute_stencil_requirements()
print("Old errorO code:")
print(vals)
print(errs)


print("Kalxulus code:")
print(num_points)
print(error)


# print(num_points, error)
# assert num_points.shape == error.shape == (kalx.x_values.size,)