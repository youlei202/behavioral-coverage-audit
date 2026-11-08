"""Portable supplementary XWVG reference-selection experiment."""
import os

# Applied before numerical imports, including in spawned solver workers.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
__version__ = '0.1.0'
