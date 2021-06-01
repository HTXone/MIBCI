import numpy as np

k = 0

Y = [[1,1],[1,1],[0,0]]
Y = np.array(Y)
#print(Y)


for i in range(1,10):
    #print(i)
    X = [[k, k + 1], [k , k + 2],[k,k+3]]
    T = np.array(X)
    k += 1
    #print(Y)
    Y = np.append(Y,T)
    #print(Y)

Y = Y.reshape((10,3,-1))
a,b,c = Y.shape
print(a)

for i in range(0,a):
    print(Y[i])
#print(Y[10])