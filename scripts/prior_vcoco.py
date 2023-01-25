##### This script will refine the predictions based on detected object by the object detector. Following by the work of https://github.com/vt-vl-lab/iCAN#######


import numpy as np
import pickle
with open('../infos/prior.pickle','rb') as fp:priors=pickle.load(fp,encoding="bytes")

def apply_prior(Object, prediction_HOI_in,predicted_single,objects_np_ex):
        prediction_HOI=np.ones(prediction_HOI_in.shape)
        return prediction_HOI

if __name__=='__main__':
    res={}
    for k in range(80):
        prediction_HOI=np.ones((1,29))
        res[k]=apply_prior([k], prediction_HOI)
