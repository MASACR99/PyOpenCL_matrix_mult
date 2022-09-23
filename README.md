# PyOpenCL_matrix_mult
A simple test project comparing CPU times (single and multithreaded) and GPU times when multiplying 2 matrixes, using PyOpenCL


## Installation
You'll need the following libraries installed via pip:
- pyopencl (pip install pyopencl) <- Please keep in mind that you may need to install other dependencies, drivers, sdks... Depending on your gpu manufacturer and OS, in my case, on Linux, I had to install intel drivers and opencl sdk and put my user into the render group.

## Personalization
You can give up to 2 integer parameters to the program to define the Matrix size (python3 main.py <size1> <size2>) if only one size is given both matrixes will be square matrixes of the specified width and height, if you put 2 values the first matrix will be of width <size1> and height <size2> and matrix2 vice-versa. If no input is given a default 2x2 matrix is used.

Matrixes are currently filled randomly by using numpy. There's no close future plans to change that.

## Future plans
Ability to provide 3 sizes and ~~CPU multithreading~~
