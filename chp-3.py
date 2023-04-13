'''def my_adder(a, b, c):
    out = a + b + c
    print(f'The value out within the function is {out}')
    return out

out = 1
d = my_adder(1, 2, 3)'''


'''def my_test(a, b):
    x = a + b
    y = x * b
    z = a + b
    
    print(f'Within function: x={x}, y={y}, z={z}')
    return x, y
a = 2
b = 3
z = 1
y, x = my_test(b, a)

print(f'Outside function: x={x}, y={y}, z={z}')'''


'''n = 42

def func():
    global n
    print(f'Within function: n is {n}')
    n = 3
    print(f'Within function: change n to {n}')

func()
print(f'Outside function: Value of n is {n}')'''


'''import numpy as np

def my_dist_xyz(x, y, z):
    """
    x, y, z are 2D coordinates contained in a tuple
    output:
    d - list, where
        d[0] is the distance between x and y
        d[1] is the distance between x and z
        d[2] is the distance between y and z
    """
    
    def my_dist(x, y):
        """
        subfunction for my_dist_xyz
        Output is the distance between x and y, 
        computed using the distance formula
        """
        out = np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)
        return out
    
    d0 = my_dist(x, y)
    d1 = my_dist(x, z)
    d2 = my_dist(y, z)
    
    return [d0, d1, d2]
d = my_dist_xyz((0, 0), (0, 1), (1, 1))
print(d)
d = my_dist((0, 0), (0, 1))'''


'''square = lambda x: x**2

print(square(2))
print(square(5))'''

'''my_adder = lambda x, y: x + y

print(my_adder(2, 4))
sorted([(1, 2), (2, 0), (4, 1)], key=lambda x: x[1])'''


'''f = max
print(type(f))'''


'''import numpy as np 

def my_fun_plus_one(f, x):
    return f(x) + 1

print(my_fun_plus_one(np.sin, np.pi/2))
print(my_fun_plus_one(np.cos, np.pi/2))
print(my_fun_plus_one(np.sqrt, 25))'''

'''import numpy as np 

def my_fun_plus_one(f, x):
    return f(x) + 1

print(my_fun_plus_one(np.sin, np.pi/2))
print(my_fun_plus_one(np.cos, np.pi/2))
print(my_fun_plus_one(np.sqrt, 25))
print(my_fun_plus_one(lambda x: x + 2, 2))'''