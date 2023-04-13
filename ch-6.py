'''def factorial(n):
    """Computes and returns the factorial of n, 
    a positive integer.
    """
    if n == 1: # Base cases!
        return 1
    else: # Recursive step
        return n * factorial(n - 1) # Recursive call     
factorial(3)'''


'''def fibonacci(n):
    """Computes and returns the Fibonacci of n, 
    a postive integer.
    """
    if n == 1: # first base case
        return 1
    elif n == 2: # second base case
        return 1
    else: # Recursive step
        return fibonacci(n-1) + fibonacci(n-2) # Recursive call 
print(fibonacci(1))
print(fibonacci(2))
print(fibonacci(3))
print(fibonacci(4))
print(fibonacci(5))'''


'''def fibonacci_display(n):
    """Computes and returns the Fibonacci of n, 
    a postive integer.
    """
    if n == 1: # first base case
        out = 1
        print(out)
        return out
    elif n == 2: # second base case
        out = 1
        print(out)
        return out
    else: # Recursive step
        out = fibonacci_display(n-1)+fibonacci_display(n-2)
        print(out)
        return out # Recursive call 
fibonacci_display(5)'''



'''def fibonacci_display(n):
    """Computes and returns the Fibonacci of n, 
    a postive integer.
    """
    if n == 1: # first base case
        out = 1
        print(out)
        return out
    elif n == 2: # second base case
        out = 1
        print(out)
        return out
    else: # Recursive step
        out = fibonacci_display(n-1)+fibonacci_display(n-2)
        print(out)
        return out # Recursive call 
fibonacci_display(35)'''
