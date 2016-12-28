# neural-network
Neural network framework in Haskell

## Packages included
- **Base**. This package defines the abstract neural-network, and a extendable
specification of layers.
- **Backend-hmatrix**. This package implements the full-connect layer and convolution
layer based on hmatrix library. It has a simple and plain representation but some issues
in both time and space performance.
- **MNIST**. Solve the MNIST with the Backend-hmatrix.
- **Backend-blashs**. This package implements the full-connect layer and convolution
layer based on blas-hs library. A imperative interface for manipulating dense vector
and matrix is devised for better storage utilization.
- **MNIST2**. Solve the MNIST2 with the Backend-blashs.


## Build with stack tool
- Please see https://docs.haskellstack.org/en/stable/README/

## Additional notes on build
### Linux
- with *openblas* flag true in the flags section, please install the openblas by the official package management.
  - or else, install blas/lapack package.

### Windows
- Download OpenBLAS from http://www.openblas.net/
- Modify the following fields in the stack.yaml
  - *extra-include-dirs:* path-to-include-dir-of-openblas
  - *extra-lib-dirs:* path-to-lib-dir-of-openblas

### Utilizing SIMD
- The *vec128* flag for *neural-network-blashs* can be turned on, and many operations will utilize SIMD for better performance.
```yaml
    neural-network-blashs:
      vec128: true
```
  - A known bug on windows. *vec128* implies compiler option **-fllvm** for ghc. However due to a known bug of binutils on mingw-w64, this option leads to a segment fault 
    - mingw-w64-x86_64-binutils < 2.27-2
    - ghc <= 8.0.1 (because it is bundled with old binutils)
    - stack resolver <= lts-7.14 (because it imples ghc <= 8.0.1)
    - See bug report here https://ghc.haskell.org/trac/ghc/ticket/8974
- The *vec256* and *vec512* cause segment-fault for the moment.
