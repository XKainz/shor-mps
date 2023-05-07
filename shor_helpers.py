'''shor_helpers.py'''
import random
import math
import contfrac

def get_x_for_N(N):
    x = random.randint(2,N)
    while math.gcd(x,N) != 1:
        x = random.randint(2,N)
    return x

def get_r(index,LargeN,N):
    value = (index,LargeN)
    convergents = list(contfrac.convergents(value))
    j = convergents[0][1]
    for i in convergents:
        if i[1] > N:
            break 
        j = i[1]
    return j

def does_r_work(x,r,N):
    if r%2 != 0:
        return 0
    g = math.gcd(int(x**(r/2)+1),N)
    if g == N or g == 1:
        return 0
    else:
        return 1
