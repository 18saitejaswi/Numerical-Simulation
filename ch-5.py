'''n = 0
for i in range(1, 4):
    n = n + i
    
print(n)'''


'''for c in "banana":
    print(c)'''
    

'''s = 0
a = [2, 3, 1, 3, 3]
for i in a:
    s += i # note this is equivalent to s = s + i
    
print(s)'''


'''dict_a = {"One":1, "Two":2, "Three":3}

for key in dict_a.keys():
    print(key, dict_a[key])'''
    
'''dict_a = {"One":1, "Two":2, "Three":3}

for key, value in dict_a.items():
    print(key, value)'''
    

'''def have_digits(s):
    
    out = 0
    
    # loop through the string
    for c in s:
        # check if the character is a digit
        if c.isdigit():
            out = 1
            break
            
    return out
out = have_digits('only4you')
print(out)'''



'''def have_digits(s):
    
    out = 0
    
    # loop through the string
    for c in s:
        # check if the character is a digit
        if c.isdigit():
            out = 1
            break
            
    return out
out = have_digits('only for you')
print(out)'''



'''import math

def my_dist_2_points(xy_points, xy):
    """
    Returns an array of distances between xy and the points 
    contained in the rows of xy_points
    
    author
    date
    """
    d = []
    for xy_point in xy_points:
        dist = math.sqrt(\
            (xy_point[0] - xy[0])**2 + (xy_point[1] - xy[1])**2)
        
        d.append(dist)
        
    return d
xy_points = [[3,2], [2, 3], [2, 2]]
xy = [1, 2]
my_dist_2_points(xy_points, xy)'''



'''x = np.array([[5, 6], [7, 8]])
n, m = x.shape
s = 0
for i in range(n):
    for j in range(m):
        s += x[i, j]
        
print(s)'''


'''i = 0
n = 8

while n >= 1:
    n /= 2
    i += 1
    
print(f'n = {n}, i = {i}')'''



'''x = range(5)
y = []

for i in x:
    y.append(i**2)
print(y)'''

'''x = range(5)
y = []

for i in x:
    y.append(i**2)
print(y)
y = [i**2 for i in x]
print(y)'''



'''y = []
for i in range(5):
    for j in range(2):
        y.append(i + j)
print(y)'''



'''y = 0
for i in range(1000):
    for j in range(1000):
        if i == j:
            y += 1'''
    