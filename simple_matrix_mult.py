import pyopencl as cl
import numpy
import time
import sys
import os

os.environ['PYOPENCL_CTX']='0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

if (len(sys.argv) > 1):
    size_x = int(sys.argv[1])
    if (len(sys.argv) > 2):
        size_y = int(sys.argv[2])
    else:
        size_y = int(sys.argv[1])
else:
    size_x = 2
    size_y = 2


matrix1 = numpy.random.rand(size_x, size_y).astype(numpy.float32)
matrix2 = numpy.random.rand(size_y, size_x).astype(numpy.float32)

print("Matrix 1 size: " + str(size_x) + ":" + str(size_y))
print("Matrix 2 size: " + str(size_y) + ":" + str(size_x))

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

print("First matrix multiplication with CPU")

if (len(matrix1[0]) != len(matrix2)):
    print("Stinky")

t0 = time.time()

matrix3 = [[0.0 for x in range(len(matrix1))] for y in range(len(matrix2[0]))]
for x in range(len(matrix3)):
    for y in range(len(matrix3[0])):
        for z in range(len(matrix1[0])):
            matrix3[x][y] += (matrix1[x][z]*matrix2[z][y])
         
t1 = time.time()

cpu_time = t1-t0

print("CPU took: " + str(cpu_time) + "s")
# print(matrix3)

print("GPU matrix multiplication begins")
# Define variables for opencl use
t0 = time.time()

mf = cl.mem_flags
matrix1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix1)
matrix2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix2)

# print(matrix1)
# print(matrix2)

kernel_params = {"w_a":size_x, "h_a": size_y, "w_b": size_x}

prg = cl.Program(ctx, """
#define WA %(w_a)d // Matrix A width
#define HA %(h_a)d // Matrix A height
#define WB %(w_b)d // Matrix B width
#define HB WA  // Matrix B height
#define WC WB  // Matrix C width
#define HC HA  // Matrix C height

__kernel void multiplication(__global const float *matrix1_g, __global const float *matrix2_g, __global float *result_matrix_g){
    int x = get_global_id(0);
    int y = get_global_id(1);

    __local float float_result;
    float_result = 0;

    for(int z = 0; z < WA;z++){
        //printf("X: %%d Y: %%d Z: %%d -> %%f & %%f\\n",x,y,z,matrix1_g[y*WA+z],matrix2_g[x+z*WB]);
        float_result += (matrix1_g[y*WA+z] * matrix2_g[x+z*WB]);
    }
    result_matrix_g[x+y*WB] = float_result;
}
"""%kernel_params).build()

result_matrix = numpy.empty((size_x, size_y)).astype(numpy.float32)
result_matrix_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_matrix.nbytes)

knl = prg.multiplication
# result_matrix.shape
event = knl(queue, result_matrix.shape, None, matrix1_g,matrix2_g,result_matrix_g)
event.wait()

cl.enqueue_copy(queue, result_matrix, result_matrix_g)
t1 = time.time()

gpu_time = t1-t0

print("GPU took: " + str(gpu_time) + "s")

# print(result_matrix)

if (cpu_time > gpu_time):
    print("GPU was faster by " + str((cpu_time*100)/gpu_time) + "%")
else:
    print("CPU was faster by " + str((gpu_time*100)/cpu_time) + "%")