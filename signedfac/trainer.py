import numpy as np
from . import sparsematrix
import tensorflow as tf
import scipy as sp
import time
import scipy.sparse
import traceback
import pickle
import threading

import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,data,model,testing_data=None,save_every=None,save_every_fn=None,
                                log_func=None):
        self.model=model
        self.double_precision=model.double_precision

        self.save_every=save_every
        if save_every is not None:
            assert save_every_fn is not None
        self.save_every_fn=save_every_fn

        self.testing_data=testing_data
        self.data=data

        if log_func is None:
            self.log_func = lambda model: None
        else:
            self.log_func=log_func

        self.logs=[]
        self.losses=[model.loss(self.data)]
        self.losses[-1]['improvement']=0
        self.losses[-1]['action']='initialization'

        self.activity_loss_crossreference=[0]
        self.activities=['initialization']
        self.activity_times=[time.time()]
        self._thread=None
        self._err=None
        self._KEEPGOING=False
        self.bestsnap=self.model.snapshot()
        self.bestsnap_activity_index=0
        self._starttime=time.time()

    def testing_loss(self,niter):
        mod=self.model.initialize_to_new_rows(self.testing_data)
        for i in range(niter):
            mod.update_rows(self.testing_data)
        return mod.loss(self.testing_data)

    def record_loss(self):
        logger.debug("grabbing loss")
        self.losses.append(self.model.loss(self.data))
        self.losses[-1]['improvement']=self.losses[-2]['loss'] - self.losses[-1]['loss']
        self.activity_loss_crossreference.append(len(self.activities)-1)
        if self.losses[-1]['loss'] <= np.min([x['loss'] for x in self.losses]):
            self.bestsnap=self.model.snapshot()
            self.bestsnap_activity_index=len(self.activities)-1
        self.logs.append(self.log_func(self.model))

    _toload=['bestsnap','logs','activities','activity_times','activity_loss_crossreference','losses','bestsnap_activity_index']

    def snapshot(self):
        dct={x:getattr(self,x) for x in self._toload}
        dct['modeltype']=self.model.__class__
        return dct

    @classmethod
    def load(cls,data,snap,save_every=None,save_every_fn=None,
                    testing_data=None,double_precision=None,log_func=None):
        self=cls.__new__(cls)
        self.model=snap['modeltype'].load(snap['bestsnap'],double_precision)

        self.save_every=save_every
        if save_every is not None:
            assert save_every_fn is not None
        self.save_every_fn=save_every_fn

        if log_func is None:
            self.log_func = lambda model: None
        else:
            self.log_func=log_func

        self.testing_data=testing_data
        self.data=data

        for nm in self._toload:
            setattr(self,nm,snap[nm])

        self._thread=None
        self._err=None
        self._KEEPGOING=False
        self._starttime=time.time()

        return self

    @property
    def is_training(self):
        return (self._thread is not None) and (self._thread.is_alive())

    @property
    def is_dead(self):
        return (self._err is not None)

    def stop_training(self):
        if not self.is_training:
            pass
        else:
            self._KEEPGOING=False
            self._thread.join()

    def save(self,force=True):
        if self.is_training:
            if force:
                pass
            else:
                raise Exception("must stop training before you save")
        snap=self.snapshot()
        dump=pickle.dumps(snap)
        with open(self.save_every_fn,'wb') as f:
            f.write(dump)


    def _record_activity(self,nm,maybe_save=True):
        self.activities.append(nm)
        self.activity_times.append(time.time())

        if maybe_save:
            if self.save_every is not None:
                if len(self.activities)%self.save_every==0:
                    self.save(force=True)
    def _perform_update(self,nm):
        logger.debug(f"running {nm}")
        getattr(self.model,'update_'+nm)(self.data)
        self._record_activity(nm)

    def restore_bestsnap(self):
        if self.is_training:
            raise Exception("Can't restore while training.")
        self.model=self.model.__class__.load(self.bestsnap,self.model.double_precision)

    def train_separate_thread(self,nms=None,maxiter=np.inf,maxtime=np.inf,freak_if_wrong=False):
        if nms is None:
            nms=self.model.things_to_update

        localstarttime=time.time()

        if self.is_training:
            raise Exception("Already training.  Call stop_training first.")
        def go():
            try:
                i=0
                while True:
                    for nm in nms:
                        if not self._KEEPGOING:
                            self._record_activity('trainstop')
                            return
                        self._perform_update(nm)
                    self._record_activity('sendweep')
                    self.record_loss()
                    if freak_if_wrong and self.losses[-1]['loss']>self.losses[-2]['loss']:
                        raise Exception("Went the wrong way!")
                    i=i+1
                    if i>maxiter:
                        return
                    if (time.time()-localstarttime)/60.0 > maxtime:
                        return
            except Exception as e:
                self._record_activity('traindeath',maybe_save=False)
                logger.warn("training thread died!")
                self._err=(e,traceback.format_exc())
                raise
        self._thread = threading.Thread(target=go)
        self._KEEPGOING=True
        self._thread.start()

    def train_tqdm_notebook(self,maxiter=1000,maxtime=np.inf,nms=None,
                    update_every_action=False,update_loss_ever=True):
        if self.is_training:
            raise Exception("Currently training; stop first.")

        if maxiter==np.inf and maxtime==np.inf:
            raise Exception("Either maxiter or maxtime must be finite.")

        if nms is None:
            nms=self.model.things_to_update

        self._err=None

        localstarttime=time.time()


        import tqdm.notebook
        trange=tqdm.notebook.tqdm(range(maxiter))
        try:
            for i in trange:
                for nm in nms:
                    descr=f"loss={self.losses[-1]['loss']:.3f} :: working on {nm}".ljust(50,'-')
                    trange.set_description(descr)
                    self._perform_update(nm)
                    if update_every_action:
                        self.record_loss()
                self._record_activity('endsweep')
                if update_loss_ever and (not update_every_action):
                    self.record_loss()
                    descr=f"loss={self.losses[-1]['loss']:.3f}".ljust(50,'-')
                    trange.set_description(descr)
                if (time.time()-localstarttime)/60.0> maxtime:
                    return
        except Exception as e:
            logger.warn("training thread died!")
            self._err=(e,traceback.format_exc())
            raise

    def status(self):
        import matplotlib.pylab as plt
        overall_losses=[x['loss'] for x in self.losses]

        if self.is_training:
            print(f"Currently training.")
            print(f"{len(self.activity_times)} updates performed.")
            print(f"{len(self.losses)} losses recorded.")
            print("Best loss so far:",np.min([x['loss'] for x in self.losses]))
        elif self.is_dead:
            print(f"We have died:")
            print(self._err[1])
        else:
            print("Not currently training.")

        if len(overall_losses)>1:
            plt.plot(overall_losses)
            plt.ylabel("loss")
