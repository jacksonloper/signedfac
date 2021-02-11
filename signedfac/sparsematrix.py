import scipy as sp
import scipy.sparse
import dataclasses
import tensorflow as tf
import numpy as np

import logging
logger = logging.getLogger(__name__)

def is_in_tensorflow_land(data):
    return hasattr(data,'_sfw') and data._sfw=='fewaklfewalkfewa'

def to_tensorflow(data,index_dtype=tf.int64,double_precision=True,store_transpose=False,consider_boolean=False):
    if double_precision:
        dtype=tf.float64
    else:
        dtype=tf.float32

    if is_in_tensorflow_land(data):
        return data
    elif sp.sparse.issparse(data): # a sparse matrix!
        return SparseMatrix.from_scipy_sparse(data,index_dtype=index_dtype,consider_boolean=consider_boolean)
    else: # dense matrix!
        data_tf=tf.convert_to_tensor(data,dtype)
        if store_transpose:
            return DenseMatrix(data,data_tf,tf.transpose(data_tf))
        else:
            return DenseMatrixNoStoredTranspose(data,data_tf,False)

@dataclasses.dataclass
class CSRMatrix:
    row_indptr: tf.Tensor
    col: tf.Tensor
    data: tf.Tensor
    shape: tuple
    nnz: int
    _sfw: str = 'fewaklfewalkfewa'


    @property
    def dtype(self):
        return self.data.dtype

    def to_scipy(self):
        return sp.sparse.csr_matrix((self.data.numpy(),self.col.numpy(),self.row_indptr.numpy()),
                        shape=self.shape)

    @classmethod
    def from_dense_tensor(cls,t,index_dtype=tf.int64):
        rows_n_cols = tf.cast(tf.where(t),dtype=index_dtype)
        data=tf.gather_nd(t,rows_n_cols)
        rows=rows_n_cols[:,0]
        cols=rows_n_cols[:,1]
        row_indptr = tf.searchsorted(rows,tf.range(0,t.shape[0]+1,dtype=rows.dtype))
        return CSRMatrix(row_indptr,cols,data,t.shape,rows.shape[0])

    @classmethod
    def from_scipy_sparse(self,data,index_dtype,consider_boolean=False):
        assert sp.sparse.issparse(data)
        dtype=data.dtype
        data=data.tocsr()
        shape=data.shape
        indptr=tf.convert_to_tensor(data.indptr,dtype=index_dtype)
        col=tf.convert_to_tensor(data.indices,dtype=index_dtype)

        if consider_boolean:
            data_tf=None
        else:
            data_tf=tf.convert_to_tensor(data.data,dtype=dtype)

        row = np.concatenate([np.ones(x)*i for (i,x) in enumerate(np.diff(indptr))])
        row = tf.convert_to_tensor(row,dtype=index_dtype)

        return CSRMatrix(indptr,col,data_tf,shape,data.nnz)

    def __getitem__(self,tp):
        if isinstance(tp,slice):
            st=tp.start
            en=tp.stop
            if st is None:
                st=0
            if en is None:
                en=self.shape[0]

            sl=slice(self.row_indptr[st],self.row_indptr[en])

            subrow=repeatrange(self.row_indptr[st+1:en+1]-self.row_indptr[st:en])
            subcol=self.col[sl]

            if self.data is None:
                subdata=tf.ones(sl.stop-sl.start,dtype=tf.float64)
            else:
                subdata=self.data[sl]

            coords=tf.stack([subrow,subcol],axis=-1)

            return tf.scatter_nd(coords,subdata,shape=(en-st,self.shape[1]))
        else:
            raise Exception("Can't slice by {type(tp)}")

@tf.function
def repeatrange(diffs):
    return tf.repeat(tf.range(len(diffs),dtype=diffs.dtype), repeats=diffs)

@dataclasses.dataclass
class DenseMatrix:
    _source: 'numpy or scipy array'
    _data: tf.Tensor
    _dataT: tf.Tensor


    _sfw: str = 'fewaklfewalkfewa'

    def __repr__(self):
        return f'<{self.shape[0]}x{self.shape[1]} DenseMatrix>'

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return DenseMatrix(self._source.T,self._dataT,self._data)

    def __getitem__(self,tp):
        if isinstance(tp,slice):
            return self._data[tp]
        elif isinstance(tp,tf.Tensor):
            return tf.gather(self._data,tp)
        else:
            raise Exception("Can't slice by {type(tp)}")

@dataclasses.dataclass
class DenseMatrixNoStoredTranspose:
    _source: 'numpy or scipy array'
    _data: tf.Tensor
    transposed: bool

    _sfw: str = 'fewaklfewalkfewa'

    def __repr__(self):
        return f'<{self.shape[0]}x{self.shape[1]} DenseMatrix>'

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return self._data.shape

    @property
    def T(self):
        return DenseMatrixNoStoredTranspose(self._source.T,self._data,not self.transposed)

    def __getitem__(self,tp):
        if self.transposed:
            if isinstance(tp,slice):
                return tf.transpose(self._data[:,tp])
            else:
                raise Exception("Can't slice by {type(tp)}")
        else:
            if isinstance(tp,slice):
                return self._data[tp]
            elif isinstance(tp,tf.Tensor):
                return tf.gather(self._data,tp)
            else:
                raise Exception("Can't slice by {type(tp)}")

@dataclasses.dataclass
class SparseMatrix:
    _source: 'numpy or scipy array'
    _sourceT: 'numpy or scipy array'
    _X_csr: CSRMatrix
    _XT_csr: CSRMatrix

    _sfw: str = 'fewaklfewalkfewa'



    def __repr__(self):
        return f'<{self.shape[0]}x{self.shape[1]} SparseMatrix with {self.nnz} entries>'

    @property
    def nnz(self):
        return self._X_csr.nnz

    @property
    def dtype(self):
        return self._X_csr.dtype

    @property
    def T(self):
        return SparseMatrix(self._sourceT,self._source,self._XT_csr,self._X_csr)

    @property
    def shape(self):
        return self._X_csr.shape

    @classmethod
    def from_scipy_sparse(cls,data,index_dtype=tf.int64,consider_boolean=False):
        sp=CSRMatrix.from_scipy_sparse(data,index_dtype,consider_boolean=consider_boolean)
        spT=CSRMatrix.from_scipy_sparse(data.T,index_dtype,consider_boolean=consider_boolean)

        return SparseMatrix(data,data.T,sp,spT)

    def __getitem__(self,tp):
        return self._X_csr[tp]
