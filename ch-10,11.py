'''def my_bad_factorial(n):
    out = 0
    for i in range(1, n+1):
        out = out*i
        
    return out
my_bad_factorial(4)'''


'''import numpy as np

s = 0
a = np.random.rand(10)
for i in range(10):
    s = s + a[i]'''
    
    

'''x = '6'
try:
    if x > 3:
        print('X is larger than 3')
except TypeError:
    print("Oops! x was not a valid number. Try again...")'''
    
    
'''x = '6'
try:
    if x > 3:
        print('X is larger than 3')
except ValueError:
    print("Oops! x was not a valid number. Try again...")'''
    
    

'''x = 's'

try:
    if x > 3:
        print(x)
except:
    print(f'Something is wrong with x = {x}')'''
    
    
    
'''def test_exceptions(x):
    try:
        x = int(x)
        if x > 3:
            print(x)
    except TypeError:
        print("Oops! x was not a valid number. Try again...")
    except ValueError:
        print("Oops! Can not convert x to integer. Try again...")
    except:
        print("Unexpected error")
x = [1, 2]
test_exceptions(x)'''



'''def test_exceptions(x):
    try:
        x = int(x)
        if x > 3:
            print(x)
    except TypeError:
        print("Oops! x was not a valid number. Try again...")
    except ValueError:
        print("Oops! Can not convert x to integer. Try again...")
    except:
        print("Unexpected error")
        x = 's'
test_exceptions(x)'''


'''x = 10

if x > 5:
    raise(Exception('x should be less or equal to 5'))'''
    
    
    
'''def my_adder(a, b, c):
    # type check
    if isinstance(a, float) and isinstance(b, float) and isinstance(c, float):
        pass
    else:
        raise(TypeError('Input arguments must be floats'))
        
    out = a + b + c
    return out
my_adder(1.0, 2.0, 3.0)'''



'''def my_adder(a, b, c):
    # type check
    if isinstance(a, float) and isinstance(b, float) and isinstance(c, float):
        pass
    else:
        raise(TypeError('Input arguments must be floats'))
        
    out = a + b + c
    return out
my_adder(1.0, 2.0, '3.0')'''



'''def my_adder(a, b, c):
    # type check
    if isinstance(a, (float, int, complex)) and isinstance(b, (float, int, complex)) and isinstance(c, (float, int, complex)):
        pass
    else:
        raise(Exception('Input arguments must be numbers'))
        
    out = a + b + c
    return out
my_adder(1, 2, 3)'''



'''def square_number(x):
    
    sq = x**2
    sq += x
    
    return sq
square_number('10')'''



'''import pdb
def square_number(x):
    
    sq = x**2
    
    # we add a breakpoint here
    pdb.set_trace()
    
    sq += x
    
    return sq
square_number(3)'''



'''f = open('test.txt', 'w')
for i in range(5):
    f.write(f"This is line {i}\n")
    
f.close()'''



'''f = open('./test.txt', 'r')
content = f.read()
f.close()
print(content)'''


'''import numpy as np
arr = np.array([[1.20, 2.20, 3.00], [4.14, 5.65, 6.42]])
arr
array([[1.2 , 2.2 , 3.  ],
       [4.14, 5.65, 6.42]])
np.savetxt('my_arr.txt', arr, fmt='%.2f', header = 'Col1 Col2 Col3')'''
