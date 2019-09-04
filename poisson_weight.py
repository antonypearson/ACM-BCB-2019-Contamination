import scipy.misc as sm
import numpy as np
from scipy.optimize import fsolve

def Wk(k,x,L):
    x = float(x)
    out = np.exp(x)
    for i in range(L):
        out-=x**i/sm.factorial(i)
    return out*x**(-k)

def intersectFG(k,p):
    L = len(p)-1
    return float(fsolve(lambda x: Wk(k,x,L)-p[L]/(sm.factorial(k)*p[k]),1))


def computeIntersections(i,P,seen,lastI):
    D = []
    for j in range(len(P)-1):
        if j!=i:
            D.append(max(lastI,(sm.factorial(i)*P[i]/(sm.factorial(j)*P[j]))**(1./(i-j))))
    D.append(intersectFG(i,P))
    maxI = max(D)
    D.insert(i,maxI+1)
    for n in seen:
        D[n] = maxI+1
    return D

def pkfk(x,k,P):
    return sm.factorial(k)*x**(-k)*np.exp(x)*P[k]
            
    
def pkfkPrime(x,k,P):
    return x**(-k)*sm.factorial(k)*np.exp(x)*(1-k/x)*P[k]



def poisson_weight(P):
    gL = True
    i = 0
    xs = [0]
    fjs = [0]
    while gL:
        intersectionsI = computeIntersections(i,P,fjs,xs[-1])
        xs.append(min(intersectionsI))
        if intersectionsI.count(min(intersectionsI))>1:
            mins = [m for m, val in enumerate(intersectionsI) if val==min(intersectionsI)]
            derivs = []
            for k in mins:
                derivs.append(pkfkPrime(min(intersectionsI),k,P))
            j = mins[derivs.index(min(derivs))]
        else:
            j = intersectionsI.index(min(intersectionsI))
        fjs.append(j)
        if j == len(P)-1:
            gL=False
        i = j
    
    potentialLambdas = []
    xs = xs[1:]
    for M in range(len(xs)):
        potentialLambdas.append(pkfk(x = xs[M],k=fjs[M],P=P))
    Lambda = max(potentialLambdas)
    alpha = xs[potentialLambdas.index(max(potentialLambdas))]
    return Lambda,alpha
