# HPC - Parallel Discrete Convolutions

A C Library, using OpenMP, for fast parallel discrete convolutions. Created by Liam Hearder (23074422) and Pranav Menon (24069351).

The formula for discrete convolutions can be found here:

$$(f*g)[n]\sum_{m=-M}^{M}f[n+m]g[m].$$

## Documentation:
### Compilation: 
There are no specific requirements for compiling our code, other than enabling recognition of OpenMP features. It can be compiled as follows:
```
gcc -fopenmp -Wall -Werror  conv2d.c -o conv2d
```

Alternatively, you could simply use the `make` command which does the same thing
___ 
### Options:
* -H `<int>` : used to set the integer height of the feature map to be generated
* -W `<int>`: used to set the integer width of the feature map to be generated
* -kH `<int>`: used to set the integer height of the kernel to be generated
* -kW `<int>`: used to the integer width of the kernel to be generated
* -f `<filepath>`: used to link the file in which the feature map is stored 
* -g `<filepath>`: used to link the file in which the kernel is stored
* -o `<filepath>`: used to provide a file in which the output will be stored
* -b: enables the benchmarking mode, which measures and outputs the performance of the program. This is a debugging flag and as such, is not required.
* -mb `<int>`: used to enable “multi-benchmarking mode” which causes the code to execute everything a number of times equal to “iterations”, then prints an average timing at the end. This is a debugging flag and as such, is not required.
* -t `<int>`: enables parallel calculation of convolutions, without which the convolutions will be calculated serially. You can optionally provide a number of threads which the application will be able to use.
___
### Sample usage:

+ With files for the kernel and feature map
    * ./conv2d -f f0.txt -g g0.txt …
+ Generating the kernel and feature map
    * ./conv2d -H 1000 -W 1000 -kH 3 -kW 3 …
+ Generating and saving the kernel and feature map
    * ./conv2d -H 1000 -W 1000 -kH 3 -kW -f feature.txt -g kernel.txt …
+ With an output file
    * ./conv2d … -o output.txt
+ Calculate in parallel with two threads
    * ./conv2d … -t 2