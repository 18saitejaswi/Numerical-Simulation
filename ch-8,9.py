'''def f(n):
    out = 0
    for i in range(n):
        for j in range(n):
            out += i*j
            
    return out'''


'''import numpy as np

def my_fib_iter1(n):
    out = np.zeros(n)
    
    out[:2] = 1
    
    for i in range(2, n):
        out[i] = out[i-1] + out[i-2]
        
    return out

def my_fib_iter2(n):
    
    out = [1, 1]
    
    for i in range(2, n):
        out.append(out[i-1]+out[i-2])
        
    return np.array(out)'''


'''def my_fib_rec(n):
    
    if n < 2:
        out = 1
    else:
        out = my_fib_rec(n-1) + my_fib_rec(n-2)
        
    return out'''


'''def my_divide_by_two(n):
    
    out = 0
    while n > 1:
        n /= 2
        out += 1
        
    return out'''



'''import numpy as np
def slow_sum(n, m):

    for i in range(n):
        # we create a size m array of random numbers
        a = np.random.rand(m)

        s = 0
        # in this loop we iterate through the array
        # and add elements to the sum one by one
        for j in range(m):
            s += a[j]   
%prun slow_sum(1000, 10000)'''



'''def add_and_subtract(iterations):
    result = 1
    
    for i in range(iterations):
        result += 1/3

    for i in range(iterations):
        result -= 1/3
    return result
# If we do this 100 times
add_and_subtract(100)'''
