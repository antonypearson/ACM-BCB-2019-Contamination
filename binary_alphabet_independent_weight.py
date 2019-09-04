import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import itertools
import scipy.optimize as so
import copy

def createIndependentSource(params):
    d = len(params)
    P = []
    Range = list(map(lambda x: ''.join(x),itertools.product(['0', '1'], repeat=d))) 
    for k in range(len(Range)):
        pk = 1.
        for j in range(d):
            pk *= (params[j]**float(Range[k][j]))*(1.-params[j])**(1.-float(Range[k][j]))
        P.append(pk)
    return P

def satisfied(s,P):
    I = [i for i in range(len(P)) if P[i]==0.]
    params = []
    for j in range(len(s)):
        if s[j] == '1':params.append(1.0)
        elif s[j] == '0':params.append(0.0)
        else:params.append(0.5)
    S = createIndependentSource(params)
    if np.all(np.array(S)[I]==np.zeros(len(I))):return True
    else:return False

def checkCompat(config,outcome):
    out = True
    for j in range(len(config)):
        if config[j]!='*' and config[j]!=outcome[j]: out = False
    return out

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]

def optimizeRegionINumHoles(I,P,config):
    #Takes 2^d vector of probs P and length-d string I (an outcome)
    variables = find(config,'*')
    M = len(variables)

    d = len(I)
    Range = list(map(lambda x: ''.join(x),itertools.product(['0', '1'], repeat=d))) 
    #Range is a sorted list of outcomes
    if M == 0:
        return P[Range.index(config)],list(map(float,list(config)))
    Kholes = [k for k in range(2**d) if P[k]==0.]
    
    cons = []
    b = (0.,np.inf)
    bnds = (b,)*M
    
    for j in range(len(Range)):
        if Range[j]!= I and j not in Kholes:# and checkCompat(config,Range[j]):#Problem possible      
            cons.append({'type': 'ineq', 'fun': lambda x,j=j:-np.prod([x[i] for i in range(M) \
                                                                       if Range[j][variables[i]]!=I[variables[i]]])+\
                        P[j]/P[Range.index(I)]})
    objective = lambda x: -P[Range.index(I)]*np.prod(x+np.ones(M)) 
    x0 = (10**-15)*np.ones(M)

    solution = so.minimize(objective,x0,method='SLSQP',\
                    bounds=bnds,constraints=cons)
    x = solution.x
    y = []
    for j in range(d):
        if j in variables:
            y.append(x[variables.index(j)]**(1-2*float(I[j]))/(1+x[variables.index(j)]**(1-2*float(I[j]))))
        else:
            y.append(float(config[j]))
    return -objective(x),y

def numerical_ind_weight(P):
    d = int(np.log2(len(P)))
    Range = list(map(lambda x: ''.join(x),itertools.product(['0', '1'], repeat=d)))
    objectives = []
    maximisers = []
    Kholes = [k for k in range(len(P)) if P[k]==0.]
    if len(Kholes) == 0:
        return NumericalMax(P)
    
    possibleConfigs = set()
    T = list(map(lambda x: ''.join(x),itertools.product(['0','1','*'],repeat=d)))
    for outcome in T:
        if satisfied(outcome,P):possibleConfigs.add(outcome)
    if len(possibleConfigs)==0:return 0.,0.
    

    for j in range(2**d):
        if j not in Kholes:
            for config in possibleConfigs:
                if checkCompat(config,Range[j]):
                    Obj,Max = optimizeRegionINumHoles(Range[j],P,config)
                    objectives.append(Obj)
                    maximisers.append(Max)
                    #print('Region: '+Range[j]+'\tConfig: '+config+   '\tSat: '+\
                     #    str(not np.any((np.array(P)-Obj*np.array(createIndependentSource(Max)))/(1-Obj)<-0.01)))
    Lambda = max(objectives)
    return Lambda, maximisers[objectives.index(Lambda)]

def DNASeq2Phat(Data):
    data = copy.deepcopy(Data)
    #Takes a list of binary strings of certain length, computes  their emprirical dist.
    for i in range(len(data)):
        data[i] = data[i].replace('\n','').upper().replace('A','0').replace('G','0').replace('C','1').replace('T','1')
    d = len(data[0])
    if d>10:return np.array([0.25,0.25,0.25,0.25])
    Astr = ['0','1']
    Range = list(itertools.product(Astr, repeat = d))
    for i in range(len(Range)):
        Range[i] = ''.join(Range[i])
    Range = sorted(Range)
    Phat = np.zeros((len(Range),1))
    for i in range(len(Range)):
        Phat[i,0] = data.count(Range[i])/float(len(data))
    return Phat
