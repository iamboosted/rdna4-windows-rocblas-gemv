# Third-party notices

The optional synthetic Q8 smoke code reconstructs the Q8_0/Q8_1 layout and
reduction arithmetic used by llama.cpp/ggml. It is included to preserve the
exact tested benchmark files, although the rocBLAS performance comparison is
FP32. The applicable ggml MIT notice is retained below. Reference checkout:
`cd959bae99ff4bed22bef2604f45241c09bbb8c8`.

The surrounding benchmark and portable helper code are distributed under the
MIT license in LICENSE. This bundle does not contain AMD SDK/runtime binaries,
rocBLAS implementation source or model weights. AMD dependencies retain their
own licenses; users provide them separately. Source links are references, not
claims of an upstream patch being incorporated into this benchmark.

## llama.cpp / ggml notice

MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
