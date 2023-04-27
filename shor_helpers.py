'''shor_helpers.py'''
import random
import math

def get_x_for_N(N):
    x = random.randint(2,N)
    while math.gcd(x,N) != 1:
        x = random.randint(2,N)
    return x