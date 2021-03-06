from numpy import *

factL = [1]

def factorial_list(n):
    # calculates all factorials up to n
    if n+1 > len(factL):
        for i in range(len(factL),n+1):
            factL.append(factL[i-1]*i)
    return factL

def Delta(j1,j2,j3):
    return factL[int(j1+j2-j3)] * factL[int(j1-j2+j3)] * factL[int(-j1+j2+j3)] *1.0 / factL[int(j1+j2+j3+1)]

def wigner3J(j1,j2,j3,m1,m2,m3):
    # calculate wigner 3j symbol using the formulas, no recursion
    # https://mathworld.wolfram.com/Wigner3j-Symbol.html
    # see also Edmonds
    
    # check some sums and triangle rules 
    if m1+m2+m3 != 0:
        return 0
    if j3 > j1 + j2:
        return 0
    if j3 < abs(j1-j2):
        return 0
    if (j1<abs(m1)) or (j2<abs(m2)) or (j3<abs(m3)):
        return 0

    # sign prefactor:
    sign = (-1)**(j1-j2-m3)
    # largest factorial needed
    maxf = int(max(j1+j2+j3+1, j1+abs(m1), j2+abs(m2), j3+abs(m3)))

    factorial_list(maxf)
    #print factL

    
    Beta  = factL[int(j1+m1)] * factL[int(j1-m1)] * factL[int(j2+m2)] * factL[int(j2-m2)] * factL[int(j3+m3)] * factL[int(j3-m3)]

    # range of t
    # t <= j1-m1, <= j2+m2, <= j1+j2-j3
    tmax = int(min(j1-m1, j2+m2, j1+j2-j3))
    # t >= j2-j3-m1, >= j1-j3+m2, >= 0
    tmin = int(max(j2-j3-m1, j1-j3+m2, 0))
    sumt = 0.0
    for t in range(tmin,tmax+1):
        x = factL[t] * factL[int(j3-j2+t+m1)] * factL[int(j3-j1+t-m2)] * factL[int(j1+j2-j3-t)] * factL[int(j1-t-m1)] * factL[int(j2-t+m2)]
        sumt = sumt + (-1.0)**t/ x

    return sign*sqrt(Delta(j1,j2,j3)*Beta)*sumt


def wigner6J(j1,j2,j3,J1,J2,J3):
    # triangle rules
    if j3 > j1 + j2:
        return 0
    if j3 < abs(j1-j2):
        return 0
    if J3 > j1 + J2:
        return 0
    if J3 < abs(j1-J2):
        return 0
    if J3 > J1 + j2:
        return 0
    if J3 < abs(J1-j2):
        return 0
    if j3 > J1 + J2:
        return 0
    if j3 < abs(J1-J2):
        return 0

    
    
    # t <= j1+j2+J1+J2, <= j2+j3+J2+J3, <= j3+j1+J3+J1
    tmax = int(min(j1+j2+J1+J2, j2+j3+J2+J3, j3+j1+J3+J1))
    # t >= j1+j2+j3, >= j1+J2+J3, >= J1+j2+J3, >= J1+J2+j3
    tmin = int(max(j1+j2+j3, j1+J2+J3, J1+j2+J3, J1+J2+j3, 0))
    #print(tmin,tmax)

    maxf = int(max(tmax+1, j1+j2+J1+J2, j2+j3+J2+J3, j3+j1+J3+J1))
    factorial_list(maxf)
    #print(factL)

    sumt = 0.0
    for t in range(tmin,tmax+1):
        x = factL[int(t-j1-j2-j3)]*factL[int(t-j1-J2-J3)]*factL[int(t-J1-j2-J3)]*factL[int(t-J1-J2-j3)] * \
            factL[int(j1+j2+J1+J2-t)] * factL[int(j2+j3+J2+J3-t)] * factL[int(j3+j1+J3+J1-t)]
        sumt = sumt + (-1.0)**t *factL[t+1]/ x

    return sqrt(Delta(j1,j2,j3)*Delta(j1,J2,J3)*Delta(J1,j2,J3)*Delta(J1,J2,j3))*sumt

def CG(j1,m1,j2,m2,j,m):
    return ((-1)**(m+j1-j2) * sqrt(2*j+1) * wigner3J(j1,j2,j,m1,m2,-m)).real

