# Upstream rocBLAS measurements: RX 9070 XT

Median of three session medians per library; each session contains six balanced-order samples per variant. Same-stream GPU microseconds per SGEMV, lower is better. See RESULTS.json for every sample and host timing.

| Rows x reduction | Layout | Retained us | Upstream us | Custom us (upstream arm) | Retained / upstream | Upstream / custom |
|---|---|---:|---:|---:|---:|---:|
| 4096 x 256 | row_major_via_transpose | 14.018 | 17.900 | 5.350 | 0.783x | 3.346x |
| 4096 x 256 | column_major_nontransposed | 3.910 | 4.113 | 11.520 | 0.951x | 0.357x |
| 65536 x 128 | row_major_via_transpose | 200.778 | 230.382 | 28.925 | 0.871x | 7.965x |
| 65536 x 128 | column_major_nontransposed | 19.508 | 19.525 | 92.935 | 0.999x | 0.210x |
| 65536 x 256 | row_major_via_transpose | 214.825 | 300.740 | 110.890 | 0.714x | 2.712x |
| 65536 x 256 | column_major_nontransposed | 108.345 | 108.375 | 212.363 | 1.000x | 0.510x |
| 65536 x 512 | row_major_via_transpose | 297.258 | 334.280 | 216.365 | 0.889x | 1.545x |
| 65536 x 512 | column_major_nontransposed | 214.477 | 213.647 | 397.293 | 1.004x | 0.538x |
| 248144 x 256 | row_major_via_transpose | 806.060 | 993.747 | 408.885 | 0.811x | 2.430x |
| 248144 x 256 | column_major_nontransposed | 406.935 | 406.618 | 482.010 | 1.001x | 0.844x |

Full BLAS correctness totals: {'numeric_reference_comparisons': 16173888, 'guard_values_checked': 23040}. Zero mismatches.

Per-case session variability is recorded in RESULTS.json. Every session is retained. Comparisons with more than 15 percent spread between session medians require qualification.

The retained package and current build can differ in build flags. The upstream build omits optional GEMM backends; native SGEMV source is unmodified. These are operation-level timings, not model inference throughput.
