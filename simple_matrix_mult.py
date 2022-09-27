import time
import sys
import os

from matrix import Matrix

os.environ['PYOPENCL_CTX']='0'
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# TODO: Check GPU multiplication and fix it when matrixes aren't squared (again)
# TODO: Create new program to run single, multi and gpu separately multiple times and 
# automatically generate a math plot with y axis being time taken and x axis being the size of the matrix (use matrix squares for simplicity)

# Check input arguments from console command
# and define sizes accordingly
if (len(sys.argv) > 1):
    size_x = int(sys.argv[1])
    if (len(sys.argv) > 2):
        size_y = int(sys.argv[2])
    else:
        size_y = int(sys.argv[1])
    
    if (len(sys.argv) > 3):
        size_y2 = int(sys.argv[3])
    else:
        size_y2 = size_x
else:
    size_x = 2
    size_y = 2
    size_y2 = 2

matrixes = Matrix(size_x,size_y,size_y2)

# Generate with specified sizes and fill with random float32 values

print("Matrix 1 size: " + str(size_x) + ":" + str(size_y))
print("Matrix 2 size: " + str(size_y) + ":" + str(size_y2) + "\n")

# If for some reason the matrixes are not multipliable, stop execution
if not matrixes.isMultipliable():
    print("Stinky")
    sys.exit(69)

print("First matrix multiplication with CPU single-core")

t0 = time.time()

result_single = matrixes.multiplySingleCore()
         
t1 = time.time()

# Save execution time
cpu_single_time = round((t1-t0),6)

print("CPU single took: " + str(cpu_single_time) + "s\n")
# Debugging/show result print
#print(matrix3)

print("CPU multithreading starts")

# Define multithreaded function
    
t0 = time.time()

result_multi = matrixes.multiplyMultiCore()

t1 = time.time()

# Debugging/show result print
# print(result_matrix_multi)

# Store execution time
cpu_multi_time = round((t1-t0),6)
print("CPU multi took: " + str(cpu_multi_time) + "s\n")

print("GPU matrix multiplication begins")

t0 = time.time()

result_gpu, time_build = matrixes.GPUMultiplication()

t1 = time.time()

# Store execution time
gpu_time = round((t1-t0),6)
time_build = round(time_build,6)

print("GPU took: " + str(gpu_time) + "s\n")
print("Separated between build time: " + str(time_build) + " and processing time: " + str(round((gpu_time-time_build),6)))

# Compare results
# Approximate results to the same decimal point to compare if all the results are equal
for i in range(len(result_single)):
    for j in range(len(result_single[0])):
        result_single[i][j] = round(result_single[i][j],2)
        result_multi[i][j] = round(result_multi[i][j],2)
        result_gpu[i][j] = round(result_gpu[i][j],2)

# Debugging/show result print
print(result_single)
print(result_gpu)
# print(result_matrix)

equals = Matrix.equals(result_single, result_multi)
    
print("CPU multicore is correct") if equals else print("CPU multicore is incorrect")

equals = Matrix.equals(result_single, result_gpu)

print("GPU multiplication is correct") if equals else print("GPU multiplication is incorrect")

# Compare execution times and get % increase in performance
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
    print("GPU was faster than CPU " + cpu_fastest_name + " by " + str(round((cpu_fastest*100)/gpu_time,2)) + "%")
else:
    print("CPU was faster than GPU by " + str(round((gpu_time*100)/cpu_fastest,2)) + "% if we count the build time")