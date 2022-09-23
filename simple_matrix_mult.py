from threading import Thread
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
print("Matrix 2 size: " + str(size_y) + ":" + str(size_x) + "\n")

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

if (len(matrix1[0]) != len(matrix2)):
    print("Stinky")
    sys.exit(69)

print("First matrix multiplication with CPU single-core")

t0 = time.time()

matrix3 = [[0.0 for x in range(len(matrix1))] for y in range(len(matrix2[0]))]
for x in range(len(matrix3)):
    for y in range(len(matrix3[0])):
        for z in range(len(matrix1[0])):
            matrix3[x][y] += (matrix1[x][z]*matrix2[z][y])
         
t1 = time.time()

cpu_single_time = t1-t0

print("CPU single took: " + str(cpu_single_time) + "s\n")
print(matrix3)

print("CPU multithreading starts")

def multithread_multiplication(matrix1, matrix2, result_matrix, starting, jump):
    x = starting
    y = 0
    while (x >= len(matrix1)):
        y += 1
        x = x - len(matrix1)
    try:
        while (x < len(matrix1[0]) and y < len(matrix1)):
            for z in range(len(matrix1[0])):
                result_matrix[x][y] += (matrix1[x][z]*matrix2[z][y])
            x += jump
            if (x >= len(matrix1)):
                y += 1
                x -= len(matrix1)
    except:
        print("Something went wrong. X:" + str(x) + "Y:" + str(y))
    
t0 = time.time()

threads = []
result_matrix_multi = [[0.0 for x in range(len(matrix1))] for y in range(len(matrix2[0]))]

if (os.cpu_count() > (size_x*size_y)):
    range_value = size_x*size_y
else:
    range_value = os.cpu_count()

for i in range(range_value):
    threads.append(Thread(target=multithread_multiplication,args=((matrix1, matrix2, result_matrix_multi,i,range_value))))

for i in range(len(threads)):
    threads[i].start()

for i in range(len(threads)):
    threads[i].join()

t1 = time.time()

# print(result_matrix_multi)

cpu_multi_time = t1-t0
print("CPU multi took: " + str(cpu_multi_time) + "s\n")

print("GPU matrix multiplication begins")
# Define variables for opencl use
t0 = time.time()

mf = cl.mem_flags
matrix1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix1)
matrix2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=matrix2)

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

print("GPU took: " + str(gpu_time) + "s\n")

print(result_matrix)

if (cpu_multi_time > cpu_single_time):
    print("CPU single core was faster than multi by " + str(round((cpu_multi_time*100)/cpu_single_time,2)) + "%")
    cpu_fastest_name = "single"
    cpu_fastest = cpu_single_time
else:
    print("CPU multi core was faster than single by " + str(round((cpu_single_time*100)/cpu_multi_time,2)) + "%")
    cpu_fastest_name = "multi"
    cpu_fastest = cpu_multi_time

print("")

if (cpu_fastest > gpu_time):
    print("GPU was faster than CPU" + cpu_fastest_name + " by " + str(round((cpu_fastest*100)/gpu_time,2)) + "%")
else:
    print("CPU was faster than CPU" + cpu_fastest_name + " by " + str(round((gpu_time*100)/cpu_fastest,2)) + "%")