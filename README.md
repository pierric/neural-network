# neural-network
Neural network framework in Haskell

## Packages included
- **Base**. This package defines the abstract neural-network, and a extendable
specification of layers.
- **Backend-hmatrix**. This package implements the full-connect layer and convolution
layer based on hmatrix library. It has a simple and plain representation but some issues
in both time and space efficiency.
- **Backend-blashs**. This package implements the full-connect layer and convolution
layer based on blas-hs library. A imperative interface for manipulating dense vector
and matrix is devised for better storage utilization.


## Build with stack tool

For example, building the package with SIMD-128
```bash
stack build --flag neural-network-blashs:vec128
```

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
- The flags *vec256* and *vec512* cause segment-fault for the moment.
