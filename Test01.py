import numpy as np
import math

#N = Number of snake robot links
""" for i1=1:N-1
    for i2 = 1:(N)
        if(i1==i2)
            D(i1,i2) = 1;
        end
        if(i2==i1+1)
            D(i1,i2) = -1;
        end
    end
end """

N = 10
matrix = np.zeros((N,N))


for i1 in range(1, N-1):
    for i2 in range(1,N):
        if i1==i2:
            matrix[i1][i2] = 1
        elif i2==i1+1:
            matrix[i1][i2] = -1
        
print(matrix)