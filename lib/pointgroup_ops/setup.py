from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

setup(
    name='PG_OP',
    ext_modules=[
        CUDAExtension('PG_OP', [
            'src/pointgroup_ops_api.cpp',
            'src/pointgroup_ops.cpp',
            'src/cuda.cu'
        ],
        include_dirs=[os.path.join(os.getenv('CONDA_PREFIX'), 'include')],
        extra_compile_args={'cxx': ['-g', '-O3'], 'nvcc': ['-O3']})
    ],
    cmdclass={'build_ext': BuildExtension}
)