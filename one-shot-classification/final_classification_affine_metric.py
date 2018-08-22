import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rng
import numpy.linalg as alg

import copy
import metric_learn

from scipy.ndimage import imread, affine_transform
from scipy.spatial.distance import cdist, pdist
from skimage.measure import block_reduce
from skimage.transform import rescale, resize, downscale_local_mean
from functools import reduce
from PIL import Image

# Parameters
nrun = 20 # number of classification runs
fname_label = 'class_labels.txt' # where class labels are stored for each run

def classification_run(folder,f_load,f_cost,ftype='cost'):
    # Compute error rate for one run of one-shot classification
    #
    # Input
    #  folder : contains images for a run of one-shot classification
    #  f_load : itemA = f_load('file.png') should read in the image file and process it
    #  f_cost : f_cost(itemA,itemB) should compute similarity between two images, using output of f_load
    #  ftype  : 'cost' if small values from f_cost mean more similar, or 'score' if large values are more similar
    #
    # Output
    #  perror : percent errors (0 to 100% error)
    # 
    assert ((ftype=='cost') | (ftype=='score'))

    # get file names
    with open(folder+'/'+fname_label) as f:
	    content = f.read().splitlines()
    pairs = [line.split() for line in content]
    test_files  = [pair[0] for pair in pairs]
    train_files = [pair[1] for pair in pairs]
    answers_files = copy.copy(train_files)
    test_files.sort()
    train_files.sort()	
    ntrain = len(train_files)
    ntest = len(test_files)

    # load the images (and, if needed, extract features)
    train_items = [f_load(f) for f in train_files]
    test_items  = [f_load(f) for f in test_files ]

    # Augment with 5 affine transforms
    # Creates 6 total examples per training item
    nexample = 6
    feat_mtx = np.zeros((nexample*ntrain,1024),dtype=float)
    for i, item in enumerate(train_items):
        I = rescale(item, 1.0 / 3.0, anti_aliasing=False)
        I = I.astype(bool)
        I = I.astype(float)
        feat_mtx[(nexample*i),:] = I.flatten()
        for j in range(1,nexample):
            feat_mtx[(nexample*i)+j,:] = AffineTransImg(item)

    # gather the class numbers for each file
    classes = np.repeat(np.arange(1,ntrain+1),nexample)

    Y = classes
    X = feat_mtx

    # setting up LMNN
    # tried 14 classes because of training data size (want 14 nearest neighbors for each class)
    # lowered to k=5 because training takes very long
    lmnn = metric_learn.LMNN(k=5, min_iter=50, max_iter=1000, learn_rate=1e-6, regularization=1)

    # fit the data!
    # Could take up to 30 minutes per run
    # Use already saved matrices to save time
    try:
        Minv = np.load(folder+'Minv.npy')
    except FileNotFoundError:
        print("Matrix file not available.\n")
        print("Fitting data...\n")
        lmnn.fit(X, Y)
    # Save Mahalanobis metric matrix as a file for later
        Minv = lmnn.metric()
        np.save(folder+'Minv.npy', Minv)
    
    

    # compute cost matrix
    costM = np.zeros((ntest,ntrain),float)
    for i in range(ntest):
	    for c in range(ntrain):
		    costM[i,c] = f_cost(test_items[i],train_items[c],Minv)
    
    if ftype == 'cost':
	    YHAT = np.argmin(costM,axis=1)
    elif ftype == 'score':
	    YHAT = np.argmax(costM,axis=1)
    else:
	    assert False

    # compute the error rate
    correct = 0.0
    for i in range(ntest):
	    if train_files[YHAT[i]] == answers_files[i]:
		    correct += 1.0
    pcorrect = 100 * correct / ntest
    perror = 100 - pcorrect
    return perror

def ModHausdorffDistance(itemA,itemB,Minv):
    # Modified Hausdorff Distance
    #
    # Input
    #  itemA : [n x 2] coordinates of "inked" pixels
    #  itemB : [m x 2] coordinates of "inked" pixels
    #
    #  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
    #  International Conference on Pattern Recognition, pp. 566-568.
    #
    itemA = rescale(itemA, 1.0 / 3.0, anti_aliasing=False).flatten()
    itemB = rescale(itemB, 1.0 / 3.0, anti_aliasing=False).flatten()
    
    items = np.stack((itemA, itemB))
    D = pdist(items, metric='mahalanobis', VI=Minv)
    return D

def LoadImgAsPoints(fn):
    # Load image file and return coordinates of 'inked' pixels in the binary image
    # 
    # Output:
    #  D : [n x 2] rows are coordinates
    I = imread(fn,flatten=True)
    I = np.array(I,dtype=bool)
    I = np.logical_not(I)

    # crop it to 96x96 for easy rescaling
    I = I[4:100,4:100]
    #I = rescale(I, 1.0 / 3.0, anti_aliasing=False)
    #I = resize(I, (I.shape[0] / 3, I.shape[1] / 3), anti_aliasing=False)
    #I = downscale_local_mean(I, (3, 3))
    #I = I.astype(bool)
    #I = I.astype(float)
    return I

def AffineTransImg(I):
    # Input: 
    # Image ndarray of floats
    #
    # Output:
    # Affine tranformed image, flattened to 1D
    theta = rng.uniform(-np.pi/18, np.pi/18)
    rhox = rng.uniform(-0.1,0.1)
    rhoy = rng.uniform(-0.1,0.1)
    sx = rng.uniform(0.9,1.1)
    sy = rng.uniform(0.9,1.1)
    tx = rng.uniform(-2,2)
    ty = rng.uniform(-2,2)
    c = np.cos(theta)
    s = np.sin(theta)
    Rot = np.array([[c,s],[-s,c]])
    Shr = np. array([[1,rhox],[rhoy,1]])
    Sca = np.array([[sx,0],[0,sy]])
    A = reduce(np.dot, [Sca, Shr, Rot])
    b = np.transpose([[0,0]])
    try:
        Ainv = alg.inv(A)
        HomoA = np.concatenate((Ainv,-np.dot(Ainv,b)),axis=1)
        HomoA = np.concatenate((HomoA,[[0,0,1]]))
        I = affine_transform(I, Ainv)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            pass
        else:
            raise     
    I = downscale_local_mean(I, (3, 3))
    I[I >= 0.5] = 1
    I[I < 0.5] = 0
    
    I = I.flatten()
    return(I)

if __name__ == "__main__":
	#
	# Running this demo should lead to a result of 38.8 percent errors.
	#
	#   M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
	#     International Conference on Pattern Recognition, pp. 566-568.
	#
	# ** Models should be trained on images in 'images_background' directory to avoid 
	#  using images and alphabets used in the one-shot evaluation **
	#
	print ('One-shot classification demo with Large Margin Nearest Neighbors')
	perror = np.zeros(nrun)
	for r in range(1,nrun+1):
		rs = str(r)
		if len(rs)==1:
			rs = '0' + rs		
		perror[r-1] = classification_run('run'+rs, LoadImgAsPoints, ModHausdorffDistance, 'cost')
		print (" run " + str(r) + " (error " + str(	perror[r-1] ) + "%)")		
	total = np.mean(perror)
	print (" average error " + str(total) + "%")
