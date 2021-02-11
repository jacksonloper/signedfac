import tensorflow as tf
import numpy as np
from . import sparsematrix


import logging
logger = logging.getLogger(__name__)


def safe_inverse_one(Lambda,numstab):
    return safe_inverse(Lambda[None],numstab)[0]

def symmetrize(Lambda):
    return .5*(Lambda+batch_transpose(Lambda))

def batch_transpose(Lambda):
    return tf.einsum('...ji->...ij',Lambda)

@tf.function(autograph=False)
def clip_low_eigenvalues(Lambda,numstab):
    with tf.device('cpu'):  # GPU eigenvalue stuff stills seem to be quite slow
        eigval,eigvec=tf.linalg.eigh(Lambda)
        eigval=tf.clip_by_value(eigval,numstab,float(np.inf))
    rez=tf.einsum('...jk,...lk,...k->...jl',eigvec,eigvec,eigval)
    rez = symmetrize(rez)
    return rez

@tf.function(autograph=False)
def safe_inverse(Lambda,numstab):
    with tf.device('cpu'):  # GPU eigenvalues still seem to be quite bad
        eigval,eigvec=tf.linalg.eigh(Lambda)
        eigval=tf.clip_by_value(eigval,numstab,float(np.inf))
        evalo = tf.linalg.diag(1.0/eigval)
        B = tf.einsum('ijk,ilk->ijl',evalo,eigvec)
    return tf.einsum('ijk,ikl->ijl',eigvec,B)

def tfify(vals,dtype=tf.float64):
    return {x:tf.convert_to_tensor(vals[x],dtype=dtype) for x in vals if (vals[x] is not None)}

def tf_variabilify(vals,dtype=tf.float64):
    return {x:tf.Variable(tf.convert_to_tensor(vals[x],dtype=dtype)) for x in vals if (vals[x] is not None)}

def tf_variabilify_one(vals,dtype=tf.float64):
    if vals is None:
        return None
    else:
        return tf.Variable(tf.convert_to_tensor(vals,dtype=dtype))

def pge_safe(x):
    switch=tf.abs(x)<.00001

    A=.25-0.020833333333333332*(x**2)
    B=tf.tanh(x/2)/(2*x)
    return tf.where(switch,A,B)

def batchup(Nr,batchsize):
    bins=np.r_[0:Nr:batchsize,Nr]
    batches=np.c_[bins[:-1],bins[1:]]
    return batches

def map_accum(fn,Nr,batchsize,rule):
    batches=batchup(Nr,batchsize)

    logger.debug(f"mapcat {fn}: 0/{Nr}")
    result=fn(batches[0,0],batches[0,1])
    assert isinstance(result,tuple),"fn should return a tuple of tensors"

    # initialize!
    if len(rule)==1:
        rule = rule * len(result)
    tas=[]
    for i,r in enumerate(result):
        if r is None:
            tas.append(None)
        elif rule[i]=='c':
            ta = tf.TensorArray(dtype=r.dtype,size=batches.shape[0],dynamic_size=False,
                            infer_shape=False,element_shape=(None,)+r.shape[1:])
            ta = ta.write(0,r)
            tas.append(ta)
        elif rule[i]=='s':
            tas.append(r)
        else:
            raise Exception(f"What is {rule[i]}?")

    # the loop!
    for i in range(1,len(batches)):
        logger.debug(f"mapcat {fn}: {batches[i,0]}/{Nr}")
        for j,r in enumerate(fn(batches[i][0],batches[i][1])):
            if tas[0] is None:
                pass
            elif rule[j]=='c':
                tas[j]=tas[j].write(i,r)
            elif rule[j]=='s':
                tas[j]=tas[j]+r
            else:
                raise Exception(f"What is {rule[j]}?")

    # close out!
    rez=[]
    for i,r in enumerate(tas):
        if tas[0] is None:
            rez.append(None)
        elif rule[i]=='c':
            rez.append(r.concat())
        elif rule[i]=='s':
            rez.append(r)
        else:
            raise Exception(f"What is {rule[j]}?")

    return tuple(rez)

def map_cat_sparse(fn,Nr,batchsize,index_dtype=tf.int64):
    CURROW=0
    WIDTH=None

    def go(st,en):
        nonlocal CURROW,WIDTH
        g=fn(st,en)

        # check for consistant widths
        if WIDTH is None:
            WIDTH=g.shape[1]
        else:
            assert WIDTH==g.shape[1]

        rows_n_cols = tf.cast(tf.where(g),dtype=index_dtype)
        data=tf.gather_nd(g,rows_n_cols)
        rows=rows_n_cols[:,0]+st
        cols=rows_n_cols[:,1]

        return rows,cols,data
    rows,cols,data = map_accum(go,Nr,batchsize,'ccc')
    row_indptr = tf.searchsorted(rows,tf.range(0,Nr+1,dtype=rows.dtype))
    return sparsematrix.CSRMatrix(row_indptr,rows,cols,data,(Nr,WIDTH),rows.shape[0])

def log2cosho2_safe(x):
    '''
    returns log(2*(cosh(x/2))
    '''

    return tf.math.softplus(x) -0.5*x

def log2cosho2_unsafe(x):
    '''
    returns log(2*(cosh(x/2))
    '''

    return tf.math.log(2*tf.cosh(x/2))

def log_binom(a,b):
    '''
    result = log a! / (b! * (a-b)!)
           = log Gamma(a+1) - log Gamma(b+1) - log Gamma(a-b+1)
    '''

    return tf.math.lgamma(a+1) - tf.math.lgamma(b+1) - tf.math.lgamma(a-b+1)


def log_negativebinomial(X,C,theta):
    '''
    log NegativeBinomial(x; p=sigmoid(c), r= theta)
    '''

    xpt=X+theta
    binc = tf.reduce_sum(log_binom(xpt-1,X))
    cosh_term = tf.reduce_sum(log2cosho2_safe(C)*xpt)
    xmh_term = .5*tf.reduce_sum(C*(X-theta))

    return xmh_term+binc-cosh_term

def inverse_digamma(X):
    '''
    find y such that digamma(y) = x

    Thomas Minka, Estimating a Dirichlet distribution,
    Technical Report 2012, Appendix C.

    by way of Baris Kurt
    '''

    M= (X>=-2.22)
    Y = tf.where(M,tf.math.exp(X)+.5,-1/(X+0.5772156649015329))

    for i in range(7):
        Y = Y - (tf.math.digamma(Y)-X)/tf.math.polygamma(1,Y);

    return Y
