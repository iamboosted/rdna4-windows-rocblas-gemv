# Upstream rocBLAS measurements

Median of three session medians per library; each session contains six balanced-order samples per variant. Same-stream GPU microseconds per SGEMV, lower is better. See RESULTS.json for every sample and host timing.

| Rows x reduction | Layout | Retained us | Upstream us | Custom us (upstream arm) | Retained / upstream | Upstream / custom |
|---|---|---:|---:|---:|---:|---:|
| 4096 x 256 | row_major_via_transpose | 61.398 | 79.498 | 18.675 | 0.772x | 4.257x |
| 4096 x 256 | column_major_nontransposed | 6.098 | 6.947 | 19.375 | 0.878x | 0.359x |
| 65536 x 128 | row_major_via_transpose | 430.858 | 523.010 | 109.047 | 0.824x | 4.796x |
| 65536 x 128 | column_major_nontransposed | 112.510 | 109.855 | 204.620 | 1.024x | 0.537x |
| 65536 x 256 | row_major_via_transpose | 454.522 | 532.543 | 217.433 | 0.853x | 2.449x |
| 65536 x 256 | column_major_nontransposed | 216.345 | 216.532 | 373.315 | 0.999x | 0.580x |
| 65536 x 512 | row_major_via_transpose | 581.632 | 667.835 | 428.788 | 0.871x | 1.557x |
| 65536 x 512 | column_major_nontransposed | 428.925 | 428.238 | 674.952 | 1.002x | 0.634x |
| 248144 x 256 | row_major_via_transpose | 1592.158 | 1985.997 | 813.642 | 0.802x | 2.441x |
| 248144 x 256 | column_major_nontransposed | 811.202 | 811.170 | 970.503 | 1.000x | 0.836x |

Full BLAS correctness totals: {'numeric_reference_comparisons': 16173888, 'guard_values_checked': 23040}. Zero mismatches.

The 4096x256 row-major case has substantial session variability in both the library and custom timings; do not use its median alone as a stable release comparison. The custom column-major 65536x128 control also varies substantially. Every session is retained, including the initially faster small case.

The retained package and current build can differ in build flags. The upstream build omits optional GEMM backends; native SGEMV source is unmodified. These are operation-level timings, not model inference throughput.
