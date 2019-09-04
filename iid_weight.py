import numpy as np
import simplex_grid as sg
import itertools


def create_iid_source(uniprob,d):
    Range = list(map(lambda x: ''.join(x),itertools.product(map(str,range(len(uniprob))), repeat=d)))
    source = []
    for x in Range:
        prob = 1.
        for i in range(len(uniprob)):
            prob *= uniprob[i]**x.count(str(i))
        source.append(prob)
    return np.array(source)

def numerical_iid_weight(P,d):
    k = int(np.log(len(P))/np.log(d))
    qs = sg.simplex_grid(k,100)/100.
    Lambda = 0.0
    Qprime = np.zeros(np.shape(P))
    for j in range(len(qs)):
        Q = create_iid_source(uniprob,d)
        if min((P/Q)[np.nonzero(Q)]):
            Lambda = min((P/Q)[np.nonzero(Q)])
            Qprime = Q

def k_less_than_five_mers2empirical_dist(data):
    empirical_dists = []
    seven_mer_indices = []

    for j in range(len(data)):
        n = len(data[j])
        d = len(data[j][0].rstrip())
        if d<=5:
            seven_mer_indices.append(j)
            empirical_dists.append([])
            for k in range(4**d):
                empirical_dists[-1].append(0.)
            Range = list(map(lambda x: ''.join(x),itertools.product(['A','C','G','T'], repeat=d)))
            for k in range(len(data[j])):
                if 'N' not in data[j][k]:
                    empirical_dists[-1][Range.index(data[j][k].rstrip())] +=1./n
    return empirical_dists,seven_mer_indices

def helper(Psumm,l):
    uppers = []
    k = len(Psumm)-1
    for i in range(l):
        uppers.append((Psumm[i]/Psumm[l])**(1.0/float(l-i)))
            
    lowers = []
    for i in range(l+1,k+1):
        lowers.append((Psumm[l]/Psumm[i])**(1.0/float(i-l)))
        
    if len(uppers) == 0:
        qLower = max(lowers)
        return [float(Psumm[l]*((1+qLower)**k)/(qLower**(k-l))), float(1.0/(1+qLower))]
    elif len(lowers) == 0:
        qUpper = min(uppers)
        return [float(Psumm[l]*((1+qUpper)**k)/(qUpper**(k-l))), float(1.0/(1+qUpper))]
    elif min(uppers)<max(lowers):
        return [0.0,0.0]
    else:
        qUpper = min(uppers)
        qLower = max(lowers)
        upp = ((1+qUpper)**k)*qUpper**(l-k)
        low = ((1+qLower)**k)*qLower**(l-k)
        if upp < low:
            return [float(Psumm[l]*low),float(1.0/(1+qLower))]
        else:
            return [float(Psumm[l]*upp),float(1.0/(1+qUpper))]

def iid_weight(P):
    d = int(np.log2(len(P)))
    if 0.0 in P:
        Q0 = constructIID2xd(0.0,d)
        Q1 = constructIID2xd(1.0,d)
        l0 = 1.0
        l1 = 1.0
        for j in range(len(Q0)):
            if Q0[j]!=0.0 and P[j]/Q0[j]<l0: l0 = P[j]/Q0[j]
            if Q1[j]!=0.0 and P[j]/Q1[j]<l1: l1 = P[j]/Q1[j]
                
        if l0>l1:
            return (l0,0.0)
        else:
            return (l1,1.0)
    xMinima = np.ones((d+1,1))
    Astr = ['0','1']
    Range = list(itertools.product(Astr, repeat = d))
    for i in range(len(Range)):
        Range[i] = ''.join(Range[i])
    
    Range = sorted(Range)
    classes = {}
    for s in Range:
        counts = [str(s.count(c)) for c in Astr]
        counts = 'delim'.join(counts)
        if counts not in classes: classes[counts] = []
        classes[counts].append(s)
    for cl in classes:      
        m = int(cl.split('delim')[1])
        for outcome in classes[cl]:
            i = Range.index(outcome)
            if P[i] < xMinima[m]: xMinima[m] = P[i]
    potentialMax = []
    for i in range(d+1):
        potentialMax.append(helper(xMinima,i))
    potentialMax = np.array(potentialMax)
    j = np.argmax(potentialMax[:,0])
    return (potentialMax[j,0],potentialMax[j,1])

def constructIID2xd(p,d):
    Astr = ['0','1']
    Range = list(itertools.product(Astr, repeat = d))
    for i in range(len(Range)):
        Range[i] = ''.join(Range[i])
    Range = sorted(Range)
    P = []
    for s in Range:
        P.append((p**s.count('1'))*(1-p)**s.count('0'))
    return P
