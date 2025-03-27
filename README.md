# CUDA_Newton_project
## Compilation

To compile the project, use the following command:

```sh
nvcc -G -o output.exe kernel.cu functions.cpp -lcublas
