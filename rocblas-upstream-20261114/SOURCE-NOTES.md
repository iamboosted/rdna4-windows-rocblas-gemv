# Current upstream source notes

These are source observations, not an independent kernel-name trace.

The [pinned SGEMV dispatch](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/projects/rocblas/library/src/blas2/rocblas_gemv_kernels.cpp#L931)
includes RDNA architecture major 12. Its FP32 transposed branch for a dimension
below 4000 uses a 256-thread block and the warp-reduction GEMV kernel.
The benchmark presents row-major M x K data as column-major K x M with transpose,
so K=128/256/512 is the reduction dimension in this branch.

The [calculation](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/projects/rocblas/library/src/blas2/gemv_device.hpp#L1247)
assigns one output column per block for dimensions below the grid cap, reduces
across the block and synchronizes before reusing shared storage. At wave32,
256 threads are eight waves per output. The experimental FP32 control uses one
32-thread wave per output and groups four outputs into a 128-thread block.
This control is specialized to the tested contiguous FP32 operation and is not
a complete replacement for the rocBLAS API.

The relevant newer [upstream change](https://github.com/ROCm/rocm-libraries/commit/e181a372d9597107173209685ce10edb0c85916a)
is the October 23 large-output grid-stride fix. Its
[grid cap](https://github.com/ROCm/rocm-libraries/blob/4e81d94c6b489ce7d37e9a7381bf2cdb4ebf7a89/projects/rocblas/library/src/include/int64_helpers.hpp)
prevents the work-item count from overflowing the launch limit. It does not
reduce this benchmark's ordinary output counts to a small occupancy-based grid.
At 256 threads the uint32-derived cap is 16,777,215 blocks, above every tested
row count. Actual performance must still be measured: the kernel body and
local build can differ from the retained binary.

Changing a library kernel is ROCm library development. It does not modify RDNA4
hardware or prove a system-wide ROCm speedup. A useful next optimization would
need broader alpha/beta, vector-stride, dimension, datatype, numerical and
device coverage before becoming a general library change.
