# HPC - Parallel Discrete Convolutions

A C Library, using OpenMP and MPI, for fast parallel discrete convolutions split across multiple processes. Created by Liam Hearder (23074422) and Pranav Menon (24069351).

The formula for discrete convolutions can be found here:

$$(f*g)[n]\sum_{m=-M}^{M}f[n+m]g[m].$$

## Documentation:
### Compilation: 
Since this project uses MPI, many MPI specific flags need to be included. Using the `mpicc` wrapper automatically provides all the necessary flags and libraries. The project also requires OpenMP, as portions of it are parallelised. It can be compiled as follows:
```
mpicc -fopenmp -Wall -Werror  conv2d.c -o conv2d
```

Alternatively, simply use the `make` command.
___ 
### Options:
* -H `<int>` : The integer height of the feature map to be generated.
* -W `<int>`: The integer width of the feature map to be generated.
* -kH `<int>`: The integer height of the kernel to be generated.
* -kW `<int>`: The integer width of the kernel to be generated.
* -sH `<int>`: The integer number of rows that will be skipped in each stride
* -sW `<int>`: The integer number of columns that will be skipped in each stride
* -f `<filepath>`: used to link the file in which the feature map is stored. If a feature map is generated, the generated values will be saved to this file.
* -g `<filepath>`: used to link the file in which the kernel is stored. If a kernel is generated, the generated values will be saved to this file.
* -o `<filepath>`: used to provide a file in which the output will be stored.
* -t `<int>`: enables parallel calculation of convolutions, without which the convolutions will be calculated serially. You can optionally provide a number of threads which the application will be able to use.
* -mb `<int>`: used to enable “multi-benchmarking mode” which causes the code to execute everything a number of times equal to “iterations”, then prints an average timing at the end. This is a debugging flag and as such, is not required.
* -b: enables the benchmarking mode, which measures and outputs the performance of the program. This is a debugging flag and as such, is not required.
___
### Sample usage:
To run this program with MPI, we'll need to use the `mpirun` command, with the `-np` flag specifying the number of processes you intend to split the work across. Since MPI automatically binds itself to a single core, we need to set the `--bind-to` flag to `none`, to ensure the parallel portions of our code still operate as expected. Otherwise, MPI+OMP execution may
not yield the expected performance improvements, as threads may be competing for the same core.
+ With files for the kernel and feature map, using 2 processes
    * mpirun -np 2 --bind-to none ./conv2d -f f0.txt -g g0.txt …
+ Generating the kernel and feature map, using 2 processes
    * mpirun -np 2 --bind-to none ./conv2d -H 1000 -W 1000 -kH 3 -kW 3 …
+ Generating and saving the kernel and feature map, using 3 processes
    * mpirun -np 3 --bind-to none ./conv2d -H 1000 -W 1000 -kH 3 -kW -f feature.txt -g kernel.txt …
+ With an output file, using 1 process
    * mpirun -np 1 --bind-to none ./conv2d … -o output.txt
+ Calculate in parallel with two threads, using 4 processes
    * mpirun -np 4 --bind-to none ./conv2d … -t 2