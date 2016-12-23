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
Please see https://docs.haskellstack.org/en/stable/README/

## Additional notes on build
### Linux
- with *openblas* flag true in the flags section, please install the openblas by the official package management.
  - or else, install blas/lapack package.
- optionally, the *vec128* flag for *neural-network-blashs* can be turned on, and many operations will utilize SIMD for better performance.
```yaml
    neural-network-blashs:
      vec128: true
```
  - *vec256* and *vec512* cause segment-fault for the moment.

### Windows
- Download OpenBLAS from http://www.openblas.net/
- Modify the following fields in the stack.yaml
  - *extra-include-dirs:* path-to-include-dir-of-openblas
  - *extra-lib-dirs:* path-to-lib-dir-of-openblas
- Do not turn on flags vec128, vec256 and vec512, for they will end in segment-fault.
