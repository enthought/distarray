cimport cython
import numpy as np

cdef inline float cabs2(float complex z):
    return z.real * z.real + z.imag * z.imag

@cython.boundscheck(False)
@cython.wraparound(False)
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

    return counts
