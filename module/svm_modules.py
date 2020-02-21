import numpy as np

def EntropySelection(probas_val, sample_num):
    e = (-probas_val * np.log2(probas_val)).sum(axis = 1)
    selection = (np.argsort(e)[::-1][:sample_num])
    selection = np.array(selection).astype(int)
    return selection
