from sklearn.metrics import log_loss, average_precision_score
import numpy as np 

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0