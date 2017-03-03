import numpy as np
from numpy.matlib import repmat
import sys
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time

# add p02 folder
sys.path.insert(0, './p02/')

get_ipython().magic(u'matplotlib inline')


print('You\'re running python %s' % sys.version.split(' ')[0])



def loaddata(filename):
    """
    Returns xTr,yTr,xTe,yTe
    xTr, xTe are in the form nxd
    yTr, yTe are in the form nx1
    """
    data = loadmat(filename)
    xTr = data["xTr"]; # load in Training data
    yTr = np.round(data["yTr"]); # load in Training labels
    xTe = data["xTe"]; # load in Testing data
    yTe = np.round(data["yTe"]); # load in Testing labels
    return xTr.T,yTr.T,xTe.T,yTe.T



def row_vectorize(x):
    return x.reshape(1,-1)


x = np.array([[1,2,5],[3,4,7]])
print x.shape
x= x.reshape(1,-1)
y = 1
print np.array([y])*x
print x.shape
print x*np.array([y])
print -1*x


def perceptronUpdate(x,y,w):
    """
    function w=perceptronUpdate(x,y,w);
    
    Implementation of Perceptron weights updating
    Input:
    x : input vector of d dimensions (1xd)
    y : corresponding label (-1 or +1)
    w : weight vector before updating
    
    Output:
    w : weight vector after updating
    """
    # just in case x, w are accidentally transposed (prevents future bugs)
    x,w = map(row_vectorize, [x,w])
    assert(y in {-1,1})
    ## fill in code here
    w = w+y*x
    return w


def perceptron(x,y):
    """
    function w=perceptron(x,y);
    
    Implementation of a Perceptron classifier
    Input:
    x : n input vectors of d dimensions (nxd)
    y : n labels (-1 or +1)
    
    Output:
    w : weight vector (1xd)
    """
    
    n, d = x.shape
    w = np.zeros(d)
    
    for c in range(100):
        m = 0
        for xi,yi in zip(x,y):
            if yi*(np.dot(w.T,xi))<=0:
                w = perceptronUpdate(xi,yi,w)
                m = m+1
        if m==0:
            break
    return w

def classifyLinear(x,w,b=0):
    """
    function preds=classifyLinear(x,w,b)
    
    Make predictions with a linear classifier
    Input:
    x : n input vectors of d dimensions (dxn)
    w : weight vector (dx1)
    b : bias (scalar)
    
    Output:
    preds: predictions (1xn)
    """
    w = w.reshape(-1)
    
    preds = np.sign(np.dot(w,x)+b)
    return preds

N = 100
# Define the symbols and colors we'll use in the plots later
symbols = ['ko', 'kx']
mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
classvals = [-1, 1]

# generate random (linarly separable) data
trainPoints = np.random.randn(N, 2) * 1.5

# defining random hyperplane
w = np.random.rand(2)

# assigning labels +1, -1 labels depending on what side of the plane they lie on
trainLabels = np.sign(np.dot(trainPoints, w))
i = np.random.permutation([i for i in range(N)])

# shuffling training points in random order
trainPoints = trainPoints[i, :]
trainLabels = trainLabels[i]

# call perceptron to find w from data
w = perceptron(trainPoints,trainLabels)
b = 0

res=300
xrange = np.linspace(-5, 5,res)
yrange = np.linspace(-5, 5,res)
pixelX = repmat(xrange, res, 1)
pixelY = repmat(yrange, res, 1).T

testPoints = np.array([pixelX.flatten(), pixelY.flatten(), np.ones(pixelX.flatten().shape)]).T
testLabels = np.dot(testPoints, np.concatenate([w.flatten(), [b]]))

Z = testLabels.reshape(res,res)
plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)
plt.scatter(trainPoints[trainLabels == classvals[0],0],
            trainPoints[trainLabels == classvals[0],1],
            marker='o',
            color='k'
           )
plt.scatter(trainPoints[trainLabels == classvals[1],0],
            trainPoints[trainLabels == classvals[1],1],
            marker='x',
            color='k'
           )
plt.quiver(0,0,w[0,0],w[0,1],linewidth=0.5, color=[0,1,0])
plt.axis('tight')
plt.show()


def binarize(x, val):
    z = np.zeros(x.shape)
    z[x != val] = 0
    z[x == val] = 1
    return z


xTr,yTr,xTe,yTe=loaddata("../resource/lib/digits.mat")
MAXITER = 10
N = 100
c = [0, 7]

ii = np.where(np.logical_or(yTr == c[0], yTr == c[1]).flatten())[0]
ii = ii[np.random.permutation([i for i in range(len(ii))])]
ii = ii[:N]

xTr = xTr[ii,:]
yTr = yTr[ii].flatten()
yTr = binarize(yTr, c[0]) * 2 - 1

n = 2    
size = 2
f, axarr = plt.subplots(1, n, sharey=True)
f.set_figwidth(size * n)
f.set_figheight(size /2 *n)

w = np.zeros(xTr[0,:].shape)
err = 1.0
# run at moast MAXITER iterations
for itr in range(MAXITER):
    for i in range(N):
        # draw offender
        axarr[1].imshow(xTr[i,:].reshape(16,16).T, cmap=plt.cm.binary_r)
        axarr[1].tick_params(axis='both', which='both', bottom='off', top='off',
                             labelbottom='off', right='off', left='off', labelleft='off')
        axarr[1].set_title('Current Sample')
            
        if classifyLinear(xTr[i,:], w) != yTr[i]:
            # do update
            w = perceptronUpdate(xTr[i,:], yTr[i], w)
            # compute new training error
            preds = classifyLinear(xTr, w)
            err = np.sum(yTr == preds) / float(len(yTr))

            # plot new vector
            axarr[0].imshow(w.reshape(16,16).T, cmap=plt.cm.binary_r)
            axarr[0].tick_params(axis='both', which='both', bottom='off', top='off',
                                 labelbottom='off', right='off', left='off', labelleft='off')
            axarr[0].set_title('Weight Vector')
            axarr[0].set_xlabel('Error: %.2f' % err)
        if err == 0.:
            break
            
    time.sleep(0.01)
    if err == 0.:
        break




