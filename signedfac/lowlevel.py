import numpy as np
import dataclasses
import tensorflow as tf
from . import helpers

has_theta={
    'bernoulli':False,
    'normal':True,
    'negativebinomial':True,
}

counts_data={
    'bernoulli':True,
    'normal':False,
    'negativebinomial':True,
}

@tf.function(autograph=False)
def get_moments(muhat_row,Sighat_row):
    return muhat_row,Sighat_row + tf.einsum('ij,ik->ijk',muhat_row,muhat_row)

@tf.function(autograph=False)
def prior_KL_ind(muhat,Sighat,mu,Sig):
    '''
    Input:
    * muhat   - n x k
    * Sighat  - n x k x k
    * mu      - n x k
    * Sig     - n x k x k

    returns sum_i^n KL(muhat[i],Sighat[i] || mu[i],Sig[i])

    Remarks:
    - The first dimension for mu can also be 1 (in which case it will be broadcast)
    - The first dimension for Sig can also be 1 (in which case it will be broadcast)
    '''

    n,k = muhat.shape
    n=tf.cast(n,dtype=muhat.dtype)
    k=tf.cast(k,dtype=muhat.dtype)

    df = muhat - mu  # n x k
    m2 = Sighat + tf.einsum('ij,ik->ijk',df,df) # n x k x k

    mahaltr = tf.reduce_sum(tf.linalg.inv(Sig)*m2)

    nobs = n*k


    sign,vals=tf.linalg.slogdet(Sighat)
    badness1 = tf.reduce_min(sign)
    sigdet_post = tf.reduce_sum(vals)

    sign,vals=tf.linalg.slogdet(Sig)
    badness2=tf.reduce_min(sign)
    sigdet_prior = n*tf.reduce_mean(vals)

    badness=tf.reduce_min(tf.concat([badness1[None],badness2[None]],axis=0))<1

    return .5 * (mahaltr - nobs + sigdet_prior - sigdet_post),badness

@tf.function(autograph=False)
def assign_stats2natural(omega1,omega2,muhat,Sighat,lr,omega1_prior=0,omega2_prior=0):
    # add in priors
    omega2=omega2 + omega2_prior
    omega1=omega1 + omega1_prior

    # new stuff!
    # new_Sighat = helpers.safe_inverse(omega2,1e-8)
    new_Sighat = tf.linalg.inv(omega2)
    new_Sighat=helpers.symmetrize(new_Sighat)
    new_muhat = tf.linalg.solve(omega2,omega1[...,None])[...,0]

    # get the result
    final_muhat = muhat*(1-lr) + new_muhat*lr
    final_Sighat = Sighat*(1-lr) + new_Sighat*lr

    # make the assignment
    muhat.assign(final_muhat)
    Sighat.assign(final_Sighat)

def accumulate_omega_for_rows(X,mom_row,mom_col,kind,theta):
    # use moments to get distribution of dot products under gaussian model
    mn,vr = mom2mnvr(mom_row,mom_col)

    # accumulate xis and then omegas
    xi1,xi2=get_xi(X,mn,vr,kind,theta) # <-- Nrow x Ncol

    if xi2.shape[0]==1 and xi2.shape[1]==1:
        omega_2r = xi2[0,0]*tf.reduce_sum(mom_col[1],axis=0)
        omega_2r = tf.tile(omega_2r[None],(X.shape[0],1,1))
    elif xi2.shape[0]==1:
        omega_2r = tf.einsum('c,cij -> ij',xi2[0],mom_col[1])
        omega_2r = tf.tile(omega_2r[None],(X.shape[0],1,1))
    elif xi2.shape[1]==1:
        omega_2r = tf.einsum('r,cij -> rij',xi2[:,0],mom_col[1])
    else:
        omega_2r = tf.einsum('rc,cij -> rij',xi2,mom_col[1])

    omega_1r = tf.einsum('rc,ci -> ri',xi1,mom_col[0])

    return omega_1r,omega_2r

def mom2mnvr(mom_row,mom_col):
    muhat_row,row_m2=mom_row
    muhat_col,col_m2=mom_col
    mn= tf.einsum('ik,jk->ij',muhat_row,muhat_col)
    e2 = tf.einsum('ajk,bjk -> ab',row_m2,col_m2)
    vr = e2-mn**2
    return mn,vr


r'''
     _ _                 _       _
  __| (_)___ _ __   __ _| |_ ___| |__
 / _` | / __| '_ \ / _` | __/ __| '_ \
| (_| | \__ \ |_) | (_| | || (__| | | |
 \__,_|_|___/ .__/ \__,_|\__\___|_| |_|
            |_|
'''

def sample(C,kind,theta):
    if kind=='normal':
        return sample_normal(C,theta)
    elif kind=='bernoulli':
        return sample_bernoulli(C,theta)
    elif kind=='negativebinomial':
        return sample_negativebinomial(C,theta)
    else:
        raise Exception("NYI")

def ELBO_dataterm(X,mn,vr,kind,theta):
    if kind=='normal':
        return ELBO_normal(X,mn,vr,theta)
    elif kind=='bernoulli':
        return ELBO_bernoulli(X,mn,vr,theta)
    elif kind=='negativebinomial':
        return ELBO_negativebinomial(X,mn,vr,theta)
    else:
        raise Exception("NYI")

def get_xi(X,mn,vr,kind,theta):
    if kind=='normal':
        return get_xi_normal(X,mn,vr,theta)
    elif kind=='bernoulli':
        return get_xi_bernoulli(X,mn,vr,theta)
    elif kind=='negativebinomial':
        return get_xi_negativebinomial(X,mn,vr,theta)
    else:
        raise Exception("NYI")

def get_zeta(X,mn,vr,kind,theta):
    if kind=='normal':
        return get_zeta_normal(X,mn,vr,theta)
    elif kind=='negativebinomial':
        return get_zeta_negativebinomial(X,mn,vr,theta)
    elif kind=='bernoulli':
        return None
    else:
        raise Exception("NYI")

def solve_theta(zetamean,kind):
    if kind=='normal':
        return solve_theta_normal(zetamean)
    elif kind=='negativebinomial':
        return solve_theta_negativebinomial(zetamean)
    elif kind=='bernoulli':
        return None
    else:
        raise Exception("NYI")

r'''
     _       _        _
  __| | __ _| |_ __ _| |_ ___ _ __ _ __ ___  ___
 / _` |/ _` | __/ _` | __/ _ \ '__| '_ ` _ \/ __|
| (_| | (_| | || (_| | ||  __/ |  | | | | | \__ \
 \__,_|\__,_|\__\__,_|\__\___|_|  |_| |_| |_|___/

'''

def ELBO_normal(X,curmu,curvar,theta):
    zeta = (X-curmu)**2 + curvar
    return -.5*zeta/theta**2 -.5*tf.math.log(np.pi*2*theta**2)

def ELBO_bernoulli(X,curmu,curvar,theta):
    EY=helpers.log2cosho2_safe(tf.math.sqrt(curmu**2+curvar))
    return (X-.5)*curmu - EY

def ELBO_negativebinomial(X,curmu,curvar,theta):
    EY=helpers.log2cosho2_safe(tf.math.sqrt(curmu**2+curvar))

    T1 = .5*(X-theta)*curmu
    T2 = (X+theta)*EY
    T3 = helpers.log_binom(X+theta-1,X)

    return T3+T1 -T2

########

def get_zeta_normal(X,curmu,curvar,theta):
    return (X-curmu)**2 + curvar

def get_zeta_negativebinomial(X,curmu,curvar,theta):
    EY=helpers.log2cosho2_safe(tf.math.sqrt(curmu**2+curvar))

    return tf.math.digamma(X+theta) - EY-.5*curmu

def solve_theta_normal(zetamean):
    return tf.sqrt(zetamean)

def solve_theta_negativebinomial(zetamean):
    return helpers.inverse_digamma(zetamean)

#########

def get_xi_normal(X,curmu,curvar,theta):
    '''
    Input
    - X      Nrow x Ncol
    - curmu  Nrow x Ncol
    - curvar Nrow x Ncol
    - theta  Nrow x Ncol  (or broadcastable to)

    Output
    - xi1    Nrow x Ncol
    - xi2    Nrow x Ncol
    '''

    return X/theta**2,1/theta**2

def get_xi_bernoulli(X,curmu,curvar,theta):
    '''
    Input
    - X      Nrow x Ncol
    - curmu  Nrow x Ncol
    - curvar Nrow x Ncol

    Output
    - xi1
    - xi2
    '''

    gamsq = curmu**2 + curvar
    gam = tf.sqrt(gamsq)

    xi2 = helpers.pge_safe(gam)
    xi1 = (X-.5)

    return xi1,xi2

def get_xi_negativebinomial(X,curmu,curvar,theta):
    '''
    Input
    - X      Nrow x Ncol
    - curmu  Nrow x Ncol
    - curvar Nrow x Ncol

    Output
    - xi1
    - xi2
    '''

    gamsq = curmu**2 + curvar
    gam = tf.sqrt(gamsq)

    xi2 = helpers.pge_safe(gam)*(X+theta)
    xi1 = .5*(X-theta)

    return xi1,xi2

#########

@tf.function(autograph=False)
def sample_normal(C,theta):
    return C + tf.random.normal(C.shape,dtype=C.dtype)*theta

@tf.function(autograph=False)
def sample_bernoulli(C,theta):
    return tf.random.uniform(C.shape,dtype=C.dtype)<tf.math.sigmoid(C)

@tf.function(autograph=False)
def sample_negativebinomial(C,theta):
    gams = tf.random.gamma([1],alpha=theta,beta=tf.math.exp(-C),dtype=C.dtype)[0]
    return tf.random.poisson([1],gams)[0]
