from . import sparsematrix
import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.sparse
from . import simplefac
import sys
import tensorflow as tf

def test_sparse():
    A=npr.randn(30,42)
    A[npr.randn(*A.shape)>0]=0
    A=sp.sparse.csr_matrix(A)
    A2=sparsematrix.SparseMatrix.from_scipy_sparse(A)
    assert np.allclose(A2[:],A.todense())
    assert np.allclose(A2[2:7],A.todense()[2:7])
    assert np.allclose(A2.T[2:7],A.todense().T[2:7])

def test_sample_sparse(kind='bernoulli'):

    model=simplefac.example_model(700,800,5,kind,mag=1,sigmultiplier=.001,dtype=tf.float32)

    tf.random.set_seed(0)
    a=model.posterior_predictive_sample(sparse=True)

    tf.random.set_seed(0)
    b=model.posterior_predictive_sample(sparse=False)

def test_simplefac(kind,Nk=2):
    model=simplefac.example_model(20,30,Nk,kind,mag=1,sigmultiplier=.01)
    data=model.posterior_predictive_sample()
    data_tf=sparsematrix.to_tensorflow(data)
    model2=simplefac.initialize(data,Nk,kind)

    def check(nm):
        loss=model2.loss(data_tf)['loss']
        getattr(model2,'update_'+nm)(data_tf)
        loss2=model2.loss(data_tf)['loss']

        assert loss2<=loss

    check('thetas')
    check('prior_cols')
    check('prior_rows')
    check('rows')
    check('cols')
