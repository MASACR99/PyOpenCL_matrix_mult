import math
import os
from threading import Thread
import time
import pyopencl as cl
import numpy

class Matrix:

    def __init__(self, size_x, size_y):
        self.matrix1 = numpy.random.rand(size_x, size_y).astype(numpy.float32)
        self.matrix2 = numpy.random.rand(size_y, size_x).astype(numpy.float32)

    def __init__(self, size_x1, size_y, size_y2):
        self.matrix1 = numpy.random.rand(size_x1, size_y).astype(numpy.float32)
        self.matrix2 = numpy.random.rand(size_y, size_y2).astype(numpy.float32)

    def getSizeX1(self):
        return len(self.matrix1)

    def getSizeX2(self):
        return len(self.matrix2)

    def getSizeY1(self):
        return len(self.matrix1[0])

    def getSizeY2(self):
        return len(self.matrix2[0])

    def isMultipliable(self):
        return (len(self.matrix1[0]) == len(self.matrix2))

    def getMatrix1(self):
        return self.matrix1

    def getMatrix2(self):
        return self.matrix2

    def getValueM1(self, x, y):
        return self.matrix1[x][y]

    def getValueM2(self, x, y):
        return self.matrix2[x][y]

    def multiplySingleCore(self):
        # Generate matrix to save results in
        matrix3 = numpy.empty((self.getSizeX1(), self.getSizeY2())).astype(numpy.float32)
        # Start single core multiplication
        for x in range(len(matrix3)):
            for y in range(len(matrix3[0])):
                # Force matrix3 to be 0, since the generation isn't necessarily 0
                matrix3[x][y] = 0
                for z in range(self.getSizeY1()):
                    matrix3[x][y] += (self.getValueM1(x,z)*self.getValueM2(z,y))

        return matrix3

    
    def __multithread_multiplication(matrix1, matrix2, result_matrix, starting, jump):
        x = 0
        y = starting
        while (y >= len(result_matrix[0])):
            x += 1
            y = y - len(result_matrix[0])
        try:
            while (x < len(result_matrix)):
                result_matrix[x][y] = 0

                for z in range(len(matrix1[0])):
                    result_matrix[x][y] += (matrix1[x][z]*matrix2[z][y])
                
                y = y + jump        
                
                while (y >= len(result_matrix[0])):
                    x += 1
                    y = y - len(result_matrix[0])
                
        except:
            print("Something went wrong. X:" + str(x) + "Y:" + str(y))

    def multiplyMultiCore(self):
        threads = []
        result_matrix_multi = numpy.empty((self.getSizeX1(), self.getSizeY2())).astype(numpy.float32)
        # Find best thread size
        if (os.cpu_count() > self.getSizeX1()**2):
            range_value = self.getSizeX1()**2
        else:
            range_value = os.cpu_count()

        print("Number of threads to generate: " + str(range_value))
        # Generate and execute threads
        for i in range(range_value):
            threads.append(Thread(target=Matrix.__multithread_multiplication,args=((self.getMatrix1(), self.getMatrix2(), result_matrix_multi,i,range_value))))

        for i in range(len(threads)):
            threads[i].start()

        for i in range(len(threads)):
            threads[i].join()
        
        return result_matrix_multi

    def GPUMultiplication(self):
        t3 = time.time()
        # Start OpenCL context to select a GPU
        ctx = cl.create_some_context()
        # and generate Command Queue to send tasks
        queue = cl.CommandQueue(ctx)
        # Define variables for opencl use
        mf = cl.mem_flags

        matrix1_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.getMatrix1())
        matrix2_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.getMatrix2())

        # Define parameters used in OpenCL code compilation/building
        kernel_params = {"w_a":self.getSizeX1(), "h_a": self.getSizeY1(), "w_b": self.getSizeY2()}

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

                for(int z = 0; z < HA;z++){
                    //printf("X: %%d Y: %%d Z: %%d -> %%f & %%f\\n",x,y,z,matrix1_g[y*WA+z],matrix2_g[x+z*WB]);
                    float_result += (matrix1_g[y*HA+z] * matrix2_g[x+z*WB]);
                }
                result_matrix_g[x+y*WB] = float_result;
            }
            """%kernel_params).build()

        t4 = time.time()
        # Generate result variable and buffer to receive result
        result_matrix = numpy.empty((self.getSizeX1(), self.getSizeY2())).astype(numpy.float32)
        for i in range(len(result_matrix)):
            for j in range(len(result_matrix[0])):
                result_matrix[i][j] = 0.0
        result_matrix_g = cl.Buffer(ctx, mf.WRITE_ONLY, result_matrix.nbytes)

        knl = prg.multiplication
        event = knl(queue, result_matrix.shape, None, matrix1_g,matrix2_g,result_matrix_g)
        event.wait()

        cl.enqueue_copy(queue, result_matrix, result_matrix_g)

        return result_matrix, t4-t3

    def equals(matrix1, matrix2):
        equals = True
        for i in range(len(matrix1)):
            if not numpy.array_equal(matrix1[i], matrix2[i]):
                equals = False
                break

        return equals