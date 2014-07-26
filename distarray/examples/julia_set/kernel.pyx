# cython: boundscheck=False
# cython: wraparound=False

cimport cython
import numpy as np

def fill_complex_plane(arr, re_ax, im_ax, resolution):
    """Fill in points on the complex coordinate plane."""
    # Drawing the coordinate plane directly like this is currently much
    # faster than trying to do it by indexing a distarray.
    # This may not be the most DistArray-thonic way to do this.

    cdef:
        float _re_ax0 = re_ax[0]
        float _im_ax0 = im_ax[0]
        float re_step = float(re_ax[1] - re_ax[0]) / resolution[0]
        float im_step = float(im_ax[1] - im_ax[0]) / resolution[1]
        unsigned int row_start, col_start, row_step, col_step, nrow, ncol, i, j, glb_i, glb_j
        float complex[:,::1] _arr = arr.ndarray

    s0, s1 = arr.distribution.global_slice
    row_start, row_step = s0.start, (s0.step or 1)
    d1 = arr.distribution[1]
    col_start, col_step = s1.start, (s1.step or 1)

    nrow = _arr.shape[0]
    ncol = _arr.shape[1]

    for i in range(nrow):
        glb_i = row_start + i * row_step
        for j in range(ncol):
            glb_j = col_start + j * col_step
            _arr[i, j].real = _re_ax0 + re_step * glb_i
            _arr[i, j].imag = _im_ax0 + im_step * glb_j


cdef inline float cabs2(float complex z):
    return z.real * z.real + z.imag * z.imag


def cython_julia_calc(float complex[:,::1] z, float complex c,
                      float z_max, unsigned int n_max):

    cdef:
        float complex zt
        float z_max2 = z_max * z_max
        unsigned int nrow, ncol, i, j, count
        unsigned int[:,::1] counts = np.zeros_like(z, dtype=np.uint32)

    nrow = z.shape[0]; ncol = z.shape[1]
    for i in range(nrow):
        for j in range(ncol):
            zt = z[i,j]
            count = 0
            while cabs2(zt) < z_max2 and count < n_max:
                zt = zt * zt + c
                count += 1
            counts[i,j] = count

    return np.asarray(counts)
