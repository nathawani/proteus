#!/bin/bash
PETSC_DIR=${PWD} PETSC_ARCH=arch-mike-opt ./configure --with-fortran-bindings=0 --prefix=${CONDA_PREFIX} --with-openmp --with-blaslapack-dir=${MKLROOT} --download-superlu --download-superlu_dist --download-superlu_dist-cmake-arguments="-DTPL_ENABLE_LAPACKLIB=ON -DXSDK_ENABLE_Fortran=OFF" --download-mumps --download-eigen --download-ctetgen --download-tetgen --download-triangle-build-exec --download-tetgen-build-exec --download-gmsh --download-hdf5 --download-hdf5-configure-arguments="--enable-parallel" --download-hypre --download-libceed --download-metis --download-parmetis --download-opencascade --download-suitesparse --download-scalapack --download-szlib --download-zlib --download-triangle --with-yaml --download-zoltan --download-kokkos --with-debugging=0