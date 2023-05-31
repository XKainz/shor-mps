'''shor_helpers.py'''
import random
import math
import contfrac
import miller_rabin
import numpy as np
import numpy_helpers as nph

def success_prob_measurement_patch(patch,x,N):
    prob_success = 0
    for i in range(len(patch)):
        if patch[i] > 1/(len(patch)*2):
            r = get_r(i,len(patch),N)
            if does_r_work(x,r,N)==1:
                prob_success += patch[i]
    return prob_success

def success_prob_measurement_samples(samples,x,N,invert=True):
    if invert == True:
        for i in range(len(samples[:,0])):
            samples[i,:]=np.flip(samples[i,:])
    successful_samples = 0
    for i in range(samples.shape[0]):
        v = samples[i,:]
        i = nph.binary_array_to_number(v)
        r = get_r(i,2**len(v),N)
        if does_r_work(x,r,N)==1:
            successful_samples += 1
    return successful_samples/samples.shape[0]


def get_x_for_N(N):
    x = random.randint(2,N)
    while math.gcd(x,N) != 1:
        x = random.randint(2,N)
    return x

def get_several_x_for_N(N,amount):
    v = []
    while len(v) < amount:
        x = get_x_for_N(N)
        if x not in v:
            v.append(x)
    return v

def get_r(index,LargeN,N):
    value = (index,LargeN)
    convergents = list(contfrac.convergents(value))
    j = convergents[0][1]
    for i in convergents:
        if i[1] > N:
            break 
        j = i[1]
    return j

def N_valid(N):
    if N%2 == 0:
        return False
    if miller_rabin.miller_rabin_deterministic32(N) == True:
        return False
    if check_N_single_prime_composite(N) == True:
        return False
    return True

def check_N_single_prime_composite(N):
    y = np.log2(N)
    for i in range(2,int(np.ceil(y))):
        u1 = int(np.ceil(2**(y/i)))
        u2 = int(np.floor(2**(y/i)))
        if u1**i == N or u2**i == N:
            return True
    return False

def does_r_work(x,r,N):
    if r%2 != 0:
        return 0
    g = math.gcd(int(x**(r/2)+1),N)
    if g == N or g == 1:
        return 0
    else:
        return 1