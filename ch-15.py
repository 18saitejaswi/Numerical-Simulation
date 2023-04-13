'''import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-poster')

%matplotlib inline

def plot_vect(x, b, xlim, ylim):
    '''
    function to plot two vectors, 
    x - the original vector
    b - the transformed vector
    xlim - the limit for x
    ylim - the limit for y
    '''
    plt.figure(figsize = (10, 6))
    plt.quiver(0,0,x[0],x[1],\
        color='k',angles='xy',\
        scale_units='xy',scale=1,\
        label='Original vector')
    plt.quiver(0,0,b[0],b[1],\
        color='g',angles='xy',\
        scale_units='xy',scale=1,\
        label ='Transformed vector')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
A = np.array([[2, 0],[0, 1]])

x = np.array([[1],[1]])
b = np.dot(A, x)
plot_vect(x,b,(0,3),(0,2))'''


'''import numpy as np
def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n
x = np.array([1, 1])
a = np.array([[0, 2], 
              [2, 3]])

for i in range(8):
    x = np.dot(a, x)
    lambda_1, x = normalize(x)
    
print('Eigenvalue:', lambda_1)
print('Eigenvector:', x)'''



'''from numpy.linalg import inv
a_inv = inv(a)

for i in range(8):
    x = np.dot(a_inv, x)
    lambda_1, x = normalize(x)
    
print('Eigenvalue:', lambda_1)
print('Eigenvector:', x)'''



'''import numpy as np
from numpy.linalg import qr
a = np.array([[0, 2], 
              [2, 3]])

q, r = qr(a)

print('Q:', q)
print('R:', r)

b = np.dot(q, r)
print('QR:', b)'''



'''import numpy as np
from numpy.linalg import qr
a = np.array([[0, 2], 
              [2, 3]])
p = [1, 5, 10, 20]
for i in range(20):
    q, r = qr(a)
    a = np.dot(r, q)
    if i+1 in p:
        print(f'Iteration {i+1}:')
        print(a)'''
        
        
        
'''import numpy as np
from numpy.linalg import eig
a = np.array([[0, 2], 
              [2, 3]])
w,v=eig(a)
print('E-value:', w)
print('E-vector', v)'''


'''import numpy as np
from numpy.linalg import eig
a = np.array([[2, 2, 4], 
              [1, 3, 5],
              [2, 3, 4]])
w,v=eig(a)
print('E-value:', w)
print('E-vector', v)'''
