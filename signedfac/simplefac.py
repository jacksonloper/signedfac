r'''
Model:

    Z_row,i ~ N(B_row C_{row,i}, Sigma_row)
    Z_col,j ~ N(B_col C_{col,i}, Sigma_col)
    C_ij = <Z_row,i,Z_col,j>
    X_ij ~ p(C_ij;theta_j)

where p can be one of
- bernoulli
- normal
- negativebinomial

'''

import dataclasses
from . import lowlevel
import tensorflow as tf
import numpy.random as npr
import numpy as np
import scipy as sp
import scipy.sparse.linalg
from . import helpers
from . import sparsematrix

import logging
logger = logging.getLogger(__name__)


class NoneArray:
    def __getitem__(self,p):
        return self

@dataclasses.dataclass
class PosteriorGaussian:
    muhat: tf.Variable
    Sighat: tf.Variable
    lr: float=1.0

    @property
    def dtype(self):
        return self.muhat.dtype

    @property
    def double_precision(self):
        return self.dtype==tf.float64

    def copy(self):
        return PosteriorGaussian(
            tf.Variable(self.muhat),
            tf.Variable(self.Sighat),
            self.lr
        )

    def validate(self):
        with tf.device('cpu'):
            return dict(
                symmetric = tf.reduce_all(self.Sighat == helpers.batch_transpose(self.Sighat)).numpy(),
                # pd = (tf.reduce_min(tf.linalg.eigvalsh(self.Sighat)[0]))==1).numpy()
                pd = tf.reduce_min(tf.linalg.eigvalsh(self.Sighat)).numpy()
            )

    @property
    def Nk(self):
        return self.muhat.shape[1]

    def improve(self,omega1,omega2,omega1_prior,omega2_prior):
        lowlevel.assign_stats2natural(omega1,omega2,self.muhat,self.Sighat,self.lr,omega1_prior,omega2_prior)

    def snapshot(self):
        return dict(muhat=self.muhat.numpy(),Sighat=self.Sighat.numpy())

    def sample(self):
        noise=tf.random.normal(self.muhat.shape,dtype=self.muhat.dtype)
        return tf.einsum('ikl,il->ik',tf.linalg.cholesky(self.Sighat),noise)+self.muhat

@dataclasses.dataclass
class PriorGaussian:
    B: tf.Variable  # Nk x Ncov
    C: tf.Variable  # Nsamp x Ncov
    Sig: tf.Variable
    sigmask: tf.Variable = None

    def __post_init__(self):
        if self.sigmask is None:
            self.sigmask=tf.Variable(tf.ones(self.Sig.shape,dtype=self.dtype))

    @property
    def dtype(self):
        return self.B.dtype

    @property
    def double_precision(self):
        return self.dtype==tf.float64

    def copy(self):
        return PriorGaussian(
            tf.Variable(self.B),
            tf.Variable(self.C),
            tf.Variable(self.Sig),
            tf.Variable(self.sigmask),
        )

    def validate(self):
        return dict(
            symmetric = tf.reduce_all(self.Sig == helpers.batch_transpose(self.Sig)).numpy(),
            # pd = (tf.linalg.slogdet(self.Sig)[0]==1).numpy()
            pd = tf.reduce_min(tf.linalg.eigvalsh(self.Sig)).numpy()
        )

    def snapshot(self):
        return dict(B=self.B.numpy(),C=self.C.numpy(),Sig=self.Sig.numpy(),sigmask=self.sigmask.numpy())

    @property
    def mu(self):
        return self.C@tf.transpose(self.B)

@dataclasses.dataclass
class GaussianMatrixVI:
    post: PosteriorGaussian
    prior: PriorGaussian

    def copy(self):
        return GaussianMatrixVI(self.post.copy(),self.prior.copy())

    def validate(self):
        return dict(
            post=self.post.validate(),
            prior=self.prior.validate()
        )

    def update_post_from_suffstats(self,omega1,omega2):
        omega2_prior=tf.linalg.inv(self.prior.Sig)
        omega1_prior = tf.einsum('kl,ik->il',omega2_prior,self.prior.mu)
        self.post.improve(omega1,omega2,omega1_prior,omega2_prior[None,:,:])

    def mult(self,k):
        k2=k[:,None]*k[None,:]
        self.prior.Sig.assign(helpers.symmetrize(self.prior.Sig*k2))
        self.prior.B.assign(self.prior.B*k[:,None])
        self.post.muhat.assign(self.post.muhat*k[None,:])
        self.post.Sighat.assign(helpers.symmetrize(self.post.Sighat*k2))

    def reinitialize_covariates(self,C):
        self.prior.C=tf.Variable(tf.convert_to_tensor(C,dtype=self.dtype))
        self.prior.B=tf.Variable(tf.zeros((self.Nk,self.prior.C.shape[1]),dtype=self.dtype))
        self.update_prior()

    def update_prior(self,numlo=1e-8):
        # get best B
        self.prior.B.assign(tf.transpose(tf.linalg.lstsq(self.prior.C,self.post.muhat)))


        # get sig
        df = self.post.muhat - self.prior.mu
        newsig = tf.reduce_mean(tf.einsum('ij,ik->ijk',df,df)+self.post.Sighat,axis=0)
        newsig = helpers.clip_low_eigenvalues(newsig,numlo)

        # zero out some bits of the sigma
        if self.prior.sigmask is not None:
            newsig=newsig * self.prior.sigmask

        self.prior.Sig.assign(newsig)

    def kl(self):
        klval,badness=lowlevel.prior_KL_ind(self.post.muhat,self.post.Sighat,self.prior.mu,self.prior.Sig)
        return klval

    def kl_subset(self,indices):
        muhat=tf.gather(self.post.muhat,indices)
        Sighat=tf.gather(self.post.Sighat,indices)
        mu=tf.gather(self.prior.mu,indices)
        klval,badness=lowlevel.prior_KL_ind(muhat,Sighat,mu,self.prior.Sig)
        return klval

    def snapshot(self):
        return dict(post=self.post.snapshot(),prior=self.prior.snapshot())

    @property
    def shape(self):
        return self.post.muhat.shape

    def shapecheck(self):
        nsamp,Nk = self.shape
        ncov= self.prior.C.shape[1]
        assert self.post.Sighat.shape==(nsamp,Nk,Nk)
        assert self.prior.C.shape==(nsamp,ncov)
        assert self.prior.B.shape==(Nk,ncov)
        assert self.prior.Sig.shape==(Nk,Nk)
        assert self.prior.sigmask.shape==(Nk,Nk)

    @property
    def Nk(self):
        return self.post.muhat.shape[1]

    @classmethod
    def load(cls,snap,dtype=tf.float64):
        return GaussianMatrixVI(
            PosteriorGaussian(**helpers.tf_variabilify(snap['post'],dtype)),
            PriorGaussian(**helpers.tf_variabilify(snap['prior'],dtype)),
        )

    @classmethod
    def load_from_muhat_and_C(cls,muhat,C,dtype=tf.float64,numlo=1e-8,
                        Sighat_ratio=.001,Sighat_adder=.001,
                        sigmask=None,diagsig=None):
        if sigmask is not None:
            assert diagsig is None
            sigmask=tf.Variable(tf.convert_to_tensor(sigmask,dtye=dtype))
        elif diagsig:
            sigmask=tf.Variable(tf.convert_to_tensor(np.eye(muhat.shape[1]),dtype=dtype))

        Sighat=(np.diag(np.var(muhat,axis=0)+Sighat_adder)[None,:,:])*np.ones(len(muhat))[:,None,None]*Sighat_ratio
        post=PosteriorGaussian(**helpers.tf_variabilify(dict(muhat=muhat,Sighat=Sighat),dtype))
        prior=PriorGaussian(
            C=tf.Variable(tf.convert_to_tensor(C,dtype=dtype)),
            B=tf.Variable(tf.zeros((post.muhat.shape[1],C.shape[1]),dtype=dtype)),
            Sig=tf.Variable(tf.zeros((post.muhat.shape[1],post.muhat.shape[1]),dtype=dtype)),
            sigmask=sigmask
        )
        gmvi=GaussianMatrixVI(post,prior)
        gmvi.shapecheck()
        gmvi.update_prior()
        return gmvi

    @property
    def dtype(self):
        return self.post.dtype

    @property
    def double_precision(self):
        return self.dtype==tf.float64

@dataclasses.dataclass
class Model:
    rowinfo: GaussianMatrixVI
    colinfo: GaussianMatrixVI
    kind: str # what is the datamodel like
    thetas: tf.Tensor # a value for each column
    row_batchsize: int = 100
    col_batchsize: int = 100

    def copy(self):
        return Model(
            self.rowinfo.copy(),
            self.colinfo.copy(),
            self.kind,
            self.thetas,
            self.row_batchsize,
            self.col_batchsize)

    @property
    def double_precision(self):
        return self.rowinfo.double_precision

    @property
    def dtype(self):
        return self.rowinfo.dtype

    def snapshot(self):
        return dict(
            rowinfo=self.rowinfo.snapshot(),
            colinfo=self.colinfo.snapshot(),
            thetas=self._snap_thetas(),
            kind=self.kind,
            row_batchsize=self.row_batchsize,
            col_batchsize=self.col_batchsize,
            double_precision=self.double_precision
        )

    def __hash__(self):
        return id(self)

    def set_lr(self,row_lr=None,col_lr=None):
        if row_lr is not None:
            self.rowinfo.post.lr=row_lr
        if col_lr is not None:
            self.colinfo.post.lr=col_lr

    @classmethod
    def _load_thetas(cls,thetas,dtype):
        return helpers.tf_variabilify_one(thetas,dtype=dtype)

    def _snap_thetas(self):
        if self.thetas is None:
            return None
        else:
            return self.thetas.numpy()

    @classmethod
    def load(cls,snap,double_precision=None):
        kwargs={}
        if 'row_batchsize' in snap:
            kwargs['row_batchsize']=snap['row_batchsize']
        if 'col_batchsize' in snap:
            kwargs['col_batchsize']=snap['col_batchsize']

        # dtype
        if double_precision is None:
            double_precision=snap['double_precision']

        if double_precision:
            dtype=tf.float64
        else:
            dtype=tf.float32

        rez=cls(
            GaussianMatrixVI.load(snap['rowinfo'],dtype=dtype),
            GaussianMatrixVI.load(snap['colinfo'],dtype=dtype),
            snap['kind'],
            cls._load_thetas(snap['thetas'],dtype),
            **kwargs
        )

        rez.rowinfo.validate()
        rez.colinfo.validate()

        return rez

    @property
    def shape(self):
        return self.rowinfo.shape[0],self.colinfo.shape[0]

    @property
    def Nk(self):
        return self.rowinfo.Nk

    @property
    def row_loadings(self):
        return self.rowinfo.post.muhat.numpy()
    @property
    def col_loadings(self):
        return self.colinfo.post.muhat.numpy()

    @property
    def nobs(self):
        shape=self.shape
        return shape[0]*shape[1]

    ############################################################
    # STAT COLLECTION


    def _mom_row(self,st=0,en=None):
        return lowlevel.get_moments(self.rowinfo.post.muhat[st:en],self.rowinfo.post.Sighat[st:en])

    def _mom_col(self,st=0,en=None):
        return lowlevel.get_moments(self.colinfo.post.muhat[st:en],self.colinfo.post.Sighat[st:en])

    def _thetas_for_rows(self):
        if lowlevel.has_theta[self.kind]:
            return self.thetas[None,:]
        else:
            return None

    def _thetas_for_cols(self,st=0,en=None):
        if lowlevel.has_theta[self.kind]:
            return self.thetas[st:en,None]
        else:
            return None

    def _get_row_data_stats(self,X):
        # collect moments about the columns
        mom_col = self._mom_col()

        # accumulate omegas, in a batched way
        def go(st,en):
            mom_row = self._mom_row(st,en)
            o1,o2=lowlevel.accumulate_omega_for_rows(
                X[st:en],mom_row,mom_col,self.kind,self._thetas_for_rows())
            return o1,o2
        return helpers.map_accum(go,self.shape[0],self.row_batchsize,'cc')

    def _get_col_data_stats(self,X):
        # collect moments about the columns
        mom_row = self._mom_row()

        # accumulate omegas, in a batched way
        def go(st,en):
            mom_col = self._mom_col(st,en)
            o1,o2=lowlevel.accumulate_omega_for_rows(
                X.T[st:en],mom_col,mom_row,self.kind,self._thetas_for_cols(st,en))
            return o1,o2
        return helpers.map_accum(go,self.shape[1],self.col_batchsize,'cc')

    def _get_zeta(self,X):
        # collect some info
        if not lowlevel.has_theta[self.kind]:
            return None
        else:
            mom_row =self._mom_row()

            def go(st,en):
                mom_col = self._mom_col(st,en)
                mn,vr=lowlevel.mom2mnvr(mom_col,mom_row)
                zeta = lowlevel.get_zeta(X.T[st:en],mn,vr,self.kind,self._thetas_for_cols(st,en))
                return tf.reduce_sum(zeta,axis=1),
            return helpers.map_accum(go,self.shape[1],self.col_batchsize,'c')[0]

    ############################################################
    # UPDATES
    things_to_update=['rows','cols','prior_rows','prior_cols','thetas','rebalance']

    def update_rebalance(self,X):
        # A=np.diag(self.rowinfo.prior.Sig.numpy())
        # B=np.diag(self.colinfo.prior.Sig.numpy())
        A=tf.reduce_mean(self.rowinfo.post.muhat**2,axis=0)
        B=tf.reduce_mean(self.colinfo.post.muhat**2,axis=0)

        k=tf.convert_to_tensor((A/B)**(.25),dtype=self.dtype)
        self.rowinfo.mult(1/k)
        self.colinfo.mult(k)

    def update_rows(self,X):
        o1r,o2r = self._get_row_data_stats(X)
        self.rowinfo.update_post_from_suffstats(o1r,o2r)

    def update_cols(self,X):
        o1c,o2c = self._get_col_data_stats(X)
        self.colinfo.update_post_from_suffstats(o1c,o2c)

    def update_thetas(self,X,minth=1e-12):
        if not lowlevel.has_theta[self.kind]:
            return
        zetamean = self._get_zeta(X) / self.shape[0]
        newthetas=lowlevel.solve_theta(zetamean,self.kind)
        newthetas=tf.clip_by_value(newthetas,minth,np.inf)
        self.thetas.assign(newthetas)

    def update_prior_rows(self,X):
        self.rowinfo.update_prior()

    def update_prior_cols(self,X):
        self.colinfo.update_prior()

    ###############################################################
    # LOSS

    def _dataterm(self,X):
        mom_col=self._mom_col()
        dataterm=0
        for (st,en) in helpers.batchup(self.shape[0],self.row_batchsize):
            logger.debug(f'dataterm: {st}/{self.shape[0]}')
            mom_row=self._mom_row(st,en)
            mn,vr=lowlevel.mom2mnvr(mom_row,mom_col)
            dataterm=dataterm+tf.reduce_sum(lowlevel.ELBO_dataterm(X[st:en],mn,vr,self.kind,self.thetas))
        return dataterm

    def _loss(self,X):
        dataterm = self._dataterm(X)
        kl_row = self.rowinfo.kl()
        kl_col = self.colinfo.kl()

        nats = -(dataterm -kl_row - kl_col) / (self.nobs)
        rez=dict(loss=nats,kl_row=kl_row,kl_col=kl_col,dataterm=dataterm)
        return rez

    def loss(self,X,calc_raw_nats=True):
        rez = self._loss(X)
        return {x:rez[x].numpy() for x in rez}

    ###############################################################
    # SAMPS

    def posterior_samples(self):
        return self.rowinfo.post.sample().numpy(),self.colinfo.post.sample().numpy()

    def posterior_predictive_sample(self,memlimit=4,sample_loadings=True,sparse=None):
        if sparse is None:
            sparse = lowlevel.counts_data[self.kind]

        if not sparse:
            if 8*self.nobs > 1e9*memlimit:
                raise Exception(f"refusing to create matrix bigger than {memlimit}GB; increase keyword memlimit or use sparse=True")

        if sample_loadings:
            Zrow,Zcol = self.rowinfo.post.sample(),self.colinfo.post.sample()
        else:
            Zrow,Zcol=self.rowinfo.post.muhat,self.colinfo.post.muhat

        # accumulate samples, in a batched way
        if sparse:
            def go(st,en):
                C=Zrow[st:en] @ tf.transpose(Zcol)
                return lowlevel.sample(C,self.kind,self._thetas_for_rows())
            samp = helpers.map_cat_sparse(go,self.shape[0],self.row_batchsize) # <-- returns CSRMatrix
            return samp.to_scipy()
        else:
            def go(st,en):
                C=Zrow[st:en] @ tf.transpose(Zcol)
                return lowlevel.sample(C,self.kind,self._thetas_for_rows()),
            samp, = helpers.map_accum(go,self.shape[0],self.row_batchsize,'c') # <-- returns Tensor
            return samp.numpy()

r'''
 _       _ _   _       _ _          _   _
(_)_ __ (_) |_(_) __ _| (_)______ _| |_(_) ___  _ __
| | '_ \| | __| |/ _` | | |_  / _` | __| |/ _ \| '_ \
| | | | | | |_| | (_| | | |/ / (_| | |_| | (_) | | | |
|_|_| |_|_|\__|_|\__,_|_|_/___\__,_|\__|_|\___/|_| |_|

                                                      '''

def initialize_half(U,C=None,dtype=tf.float64,sigmultiplier=.1,diagsig=False):
    Sig = np.diag(np.var(U,axis=0))

    if C is None:
        C=tf.ones((U.shape[0],1),dtype=dtype)

    Nk=U.shape[1]

    if diagsig:
        sigmask= tf.Variable(tf.eye(Nk,dtype=dtype))
    else:
        sigmask = None

    gmvi=GaussianMatrixVI(
        post = PosteriorGaussian(
            muhat = tf.Variable(tf.convert_to_tensor(U,dtype=dtype)),
            Sighat = tf.Variable(tf.convert_to_tensor((sigmultiplier**2)*np.tile(Sig,[U.shape[0],1,1])*np.std(U),dtype=dtype))
        ),
        prior = PriorGaussian(
            C = tf.Variable(C),
            B = tf.Variable(tf.zeros((U.shape[1],C.shape[1]),dtype=dtype)),
            Sig = tf.Variable(tf.convert_to_tensor(np.cov(U.T).reshape((Nk,Nk)),dtype=dtype)),
            sigmask=sigmask,
        )
    )
    gmvi.update_prior()
    return gmvi

def make_dmhalf_op(data):
    if sp.sparse.issparse(data):
        ASL=sp.sparse.linalg.aslinearoperator
        assert sp.sparse.issparse(data)
        Nr,Nc = data.shape
        one_Nr=ASL(np.ones((Nr,1)))
        one_Nc=ASL(np.ones((Nc,1)))
        return 4*(ASL(data) - .5*(one_Nr@one_Nc.H))
    else:
        return 4*(data-.5)

def make_log_op(data):
    if sp.sparse.issparse(data):
        data=data.tocsr()
        data=sp.sparse.csr_matrix((np.log(1+data.data),data.indices,data.indptr))
        return data
    else:
        return np.log(1+data)

def var_data(data):
    if sp.sparse.issparse(data):
        c = data.copy()
        c.data **= 2
        m2 = np.asarray(np.mean(c,axis=0)).ravel()
        m1 = np.asarray(np.mean(data,axis=0)).ravel()
        return m2-m1**2
    else:
        return np.var(data,axis=0)

def initialize_with_new_rows(n_rows,snap,covariates=None):
    model=Model.load(snap)
    Nk=model.Nk

    if covariates is None:
        model.rowinfo.prior.C=tf.Variable(tf.ones((n_rows,1),dtype=model.dtype))
        model.rowinfo.prior.B=tf.Variable(tf.zeros((model.Nk,1),dtype=model.dtype))
    else:
        model.rowinfo.prior.C=tf.Variable(tf.convert_to_tensor(covariates,dtype=model.dtype))
        model.rowinfo.prior.B=tf.Variable(tf.zeros((model.Nk,covariates.shape[1]),dtype=model.dtype))

    model.rowinfo.post.Sighat=tf.Variable(tf.zeros((n_rows,Nk,Nk),dtype=model.dtype))
    model.rowinfo.post.muhat=tf.Variable(tf.zeros((n_rows,Nk),dtype=model.dtype))
    model.rowinfo.post.Sighat.assign(
        model.rowinfo.prior.Sig[None,:,:]*tf.ones(n_rows,dtype=model.dtype)[:,None,None])

    return model

def initialize(data,Nk,kind,dtype=tf.float64,diagsig=False,row_covariates=None,col_covariates=None,
                                minvar=1e-8):
    if sparsematrix.is_in_tensorflow_land(data):
        data=data._source.astype(np.float64)
    else:
        data=data.astype(np.float64)

    if kind=='normal':
        thetas=var_data(data)
        thetas[thetas<minvar]=minvar
        thetas=tf.convert_to_tensor(thetas,dtype=dtype)
        tot=data
    elif kind=='bernoulli':
        tot= make_dmhalf_op(data)
        thetas= tf.zeros(data.shape[1],dtype=dtype)
    elif kind=='negativebinomial':
        thetas=tf.ones(data.shape[1],dtype=dtype)
        tot = make_log_op(data)
    else:
        raise Exception("NYI")

    U,e,V=sp.sparse.linalg.svds(tot,Nk)
    V=V.T
    U=U@np.diag(np.sqrt(e))
    V=V@np.diag(np.sqrt(e))

    model= Model(
        initialize_half(U,dtype=dtype,diagsig=diagsig),
        initialize_half(V,dtype=dtype,diagsig=diagsig),
        thetas=tf.Variable(thetas),
        kind=kind
    )

    if row_covariates is not None:
        model.rowinfo.reinitialize_covariates(row_covariates)
    if col_covariates is not None:
        model.colinfo.reinitialize_covariates(col_covariates)

    return model


def example_model(N_rows,N_cols,Nk,kind,mag=1,sigmultiplier=.1,dtype=tf.float64):
    return Model(
        initialize_half(npr.randn(N_rows,Nk)*np.sqrt(mag),sigmultiplier=sigmultiplier,dtype=dtype),
        initialize_half(npr.randn(N_cols,Nk)*np.sqrt(mag),sigmultiplier=sigmultiplier,dtype=dtype),
        thetas=tf.Variable(tf.convert_to_tensor(1+npr.rand(N_cols)*.05,dtype=dtype)),
        kind=kind,
    )
