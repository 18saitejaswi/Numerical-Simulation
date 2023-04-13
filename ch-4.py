'''def my_thermo_stat(temp, desired_temp):
    """
    Changes the status of the thermostat based on 
    temperature and desired temperature
    author
    date
    :type temp: Int
    :type desiredTemp: Int
    :rtype: String
    """
    if temp < desired_temp - 5:
        status = 'Heat'
    elif temp > desired_temp + 5:
        status = 'AC'
    else:
        status = 'off'
    return status
status = my_thermo_stat(65,75)
print(status)'''


'''def my_thermo_stat(temp, desired_temp):
    """
    Changes the status of the thermostat based on 
    temperature and desired temperature
    author
    date
    :type temp: Int
    :type desiredTemp: Int
    :rtype: String
    """
    if temp < desired_temp - 5:
        status = 'Heat'
    elif temp > desired_temp + 5:
        status = 'AC'
    else:
        status = 'off'
    return status
status = my_thermo_stat(75,65)
print(status)'''


'''def my_thermo_stat(temp, desired_temp):
    """
    Changes the status of the thermostat based on 
    temperature and desired temperature
    author
    date
    :type temp: Int
    :type desiredTemp: Int
    :rtype: String
    """
    if temp < desired_temp - 5:
        status = 'Heat'
    elif temp > desired_temp + 5:
        status = 'AC'
    else:
        status = 'off'
    return status
status = my_thermo_stat(65,63)
print(status)'''


'''x = 3
if x > 1:
    y = 2
elif x > 2:
    y = 4
else:
    y = 0
print(y)'''


'''x = 3
if x > 1 and x < 2:
    y = 2
elif x > 2 and x < 4:
    y = 4
else:
    y = 0
print(y)'''


'''x = 3
if 1 < x < 2:
    y = 2
elif 2 < x < 4:
    y = 4
else:
    y = 0
print(y)'''


'''def my_nested_branching(x,y):
    """
    Nested Branching Statement Example
    author
    date
    :type x: Int
    :type y: Int
    :rtype: Int
    """
    if x > 2:
        if y < 2:
            out = x + y
        else:
            out = x - y
    else:
        if y > 2:
            out = x*y
        else:
            out = 0
    return out
all([1, 1, 0])'''


'''def my_adder(a, b, c):
    """
    Calculate the sum of three numbers
    author
    date
    """
    
    # Check for erroneous input
    if not (isinstance(a, (int, float)) \
            or isinstance(b, (int, float)) \
            or isinstance(c, (int, float))):
        raise TypeError('Inputs must be numbers.')
    # Return output
    return a + b + c
x = my_adder(1,2,3)
print(x)'''


'''def is_odd(number):
    """
    function returns 'odd' if the input is odd, 
       'even' otherwise
    author
    date
    :type number: Int
    :rtype: String
    """
    # use modulo to check if the input is divisible by 2
    if number % 2 == 0:
        # if it is divisible by 2, then input is not odd
        return 'even'
    else:
        return 'odd'
is_odd(11)'''



'''is_student = True
person = 'student' if is_student else 'not student'
print(person)'''


'''is_student = True
if is_student:
    person = 'student'
else:
    person = 'not student'
print(person)'''