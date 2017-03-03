
import numpy as np
import sys

sys.path.insert(0, './p03/')
get_ipython().magic(u'matplotlib inline')


import numpy as np
def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v


def name2features(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X

def genTrainFeatures(dimension=128, fix=3):

    Xgirls = name2features("C:\Users\sindh\Downloads\cornell\CourseWork\ML\Assignment3\girls.train", B=dimension, FIX=fix)
    Xboys = name2features("boys.train", B=dimension, FIX=fix)
    X = np.concatenate([Xgirls, Xboys])
    
    # Generate Labels
    Y = np.concatenate([-np.ones(len(Xgirls)), np.ones(len(Xboys))])
    
    # shuffle data into random order
    ii = np.random.permutation([i for i in range(len(Y))])
    
    return X[ii, :], Y[ii]


X,Y = genTrainFeatures(128)

def naivebayesPY(x,y):
    """
    function [pos,neg] = naivebayesPY(x,y);

    Computation of P(Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)

    Output:
    pos: probability p(y=1)
    neg: probability p(y=-1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    y = np.concatenate([y, [-1,1]])
    n = len(y)
    ## fill in code here
    p = float(np.bincount(y==1)[0])
    ne = float(np.bincount(y==-1)[0])
    pos = p/n
    neg = ne/n
    print p,ne
    return pos,neg

pos,neg = naivebayesPY(X,Y)
print pos,neg



def naivebayesPXY(x,y):
    """
    function [posprob,negprob] = naivebayesPXY(x,y);
    
    Computation of P(X|Y)
    Input:
        x : n input vectors of d dimensions (nxd)
        y : n labels (-1 or +1) (nx1)
    
    Output:
    posprob: probability vector of p(x|y=1) (dx1)
    negprob: probability vector of p(x|y=-1) (dx1)
    """
    
    # add one positive and negative example to avoid division by zero ("plus-one smoothing")
    n, d = x.shape
    x = np.concatenate([x, np.ones((2,d))])
    y = np.concatenate([y, [-1,1]])
    n, d = x.shape
    print x.shape
    y = np.array(y)
    
    xnew = np.column_stack((x,y))

    boys = xnew[xnew[:,-1]==1][:,:-1]
    girls = xnew[xnew[:,-1]==-1][:,:-1]
    
    boysVectorSum = np.sum(boys,axis=0)
    girlsVectorSum = np.sum(girls,axis=0)
    negprob= girlsVectorSum/float(sum(girlsVectorSum))
    posprob = boysVectorSum/float(sum(boysVectorSum))
    
    return posprob,negprob
    

    
posprob,negprob = naivebayesPXY(X,Y)
print posprob,negprob





def naivebayes(x,y,xtest):
    """
    function logratio = naivebayes(x,y);
    
    Computation of log P(Y|X=x1) using Bayes Rule
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    xtest: input vector of d dimensions (1xd)
    
    Output:
    logratio: log (P(Y = 1|X=x1)/P(Y=-1|X=x1))
    """
    
    ## fill in code here
    posprob,negprob = naivebayesPXY(X,Y)
    pos,neg = naivebayesPY(X,Y)
    p = -(sum(np.log(negprob)))-neg+(sum(np.log(posprob)))+pos


p = naivebayes(X,Y,X[0,:])

def naivebayesCL(x,y):
    """
    function [w,b]=naivebayesCL(x,y);
    Implementation of a Naive Bayes classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)

    Output:
    w : weight vector of d dimensions
    b : bias (scalar)
    """
    
    n, d = x.shape
    ## fill in code here
    
    posprob,negprob = naivebayesPXY(X,Y)
    pos,neg = naivebayesPY(X,Y)
    w = np.log(posprob)-np.log(negprob)
    b = np.log(pos)-np.log(neg)
    return w,b



w,b = naivebayesCL(X,Y)

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b);
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    w : weight vector of d dimensions
    b : bias (optional)
    
    Output:
    preds: predictions
    """
    pred = np.dot(x,w.T) +b
    pred[pred[:]>0]=1
    pred[pred[:]<0]=-1
    return pred

print('Training error: %.2f%%' % (100 *(classifyLinear(X, w, b) != Y).mean()))


# You can now test your code with the following interactive name classification script:

# In[ ]:

DIMS = 128
print('Loading data ...')
X,Y = genTrainFeatures(DIMS)
print('Training classifier ...')
w,b=naivebayesCL(X,Y)
error = np.mean(classifyLinear(X,w,b) != Y)
print('Training error: %.2f%%' % (100 * error))

while True:
    print('Please enter your name>')
    yourname = input()
    if len(yourname) < 1:
        break
    xtest = name2features(yourname,B=DIMS,LoadFile=False)
    pred = classifyLinear(xtest,w,b)[0]
    if pred > 0:
        print("%s, I am sure you are a nice boy.\n" % yourname)
    else:
        print("%s, I am sure you are a nice girl.\n" % yourname)

def hashfeatures(baby, B, FIX):
    v = np.zeros(B)
    for m in range(FIX):
        featurestring = "prefix" + baby[:m]
        v[hash(featurestring) % B] = 1
        featurestring = "suffix" + baby[-1*m:]
        v[hash(featurestring) % B] = 1
    return v

def name2features2(filename, B=128, FIX=3, LoadFile=True):
    """
    Output:
    X : n feature vectors of dimension B, (nxB)
    """
    # read in baby names
    if LoadFile:
        with open(filename, 'r') as f:
            babynames = [x.rstrip() for x in f.readlines() if len(x) > 0]
    else:
        babynames = filename.split('\n')
    n = len(babynames)
    X = np.zeros((n, B))
    for i in range(n):
        X[i,:] = hashfeatures(babynames[i], B, FIX)
    return X



