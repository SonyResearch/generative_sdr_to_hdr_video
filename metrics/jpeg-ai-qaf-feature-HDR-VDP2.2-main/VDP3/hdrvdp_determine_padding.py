import numpy as np
from VDP3.geomean import geomean
def hdrvdp_determine_padding(reference, metric_par):
    if isinstance(metric_par.surround, str):
        if metric_par.surround == 'none':
            pad_value = 'symmetric'
        elif metric_par.surround == 'mean':
            pad_value = geomean(np.reshape(reference, (reference.shape[0]*reference.shape[1], reference.shape[2])))
        else:
            raise ValueError('Unrecognized "surround" setting')
    else:
        if len(metric_par.surround) == 1:
            pad_value = np.repmat(metric_par.surround, [1, reference.shape[2]])
        elif len(metric_par.surround) == reference.shape[2]:
            metric_par.pad_value = metric_par.surround
        else:
            raise ValueError('The length of the "surround" vector should be 1 or equal to the number of channels')
    return pad_value
