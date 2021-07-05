import ctypes
from utils import load_lib, c_float_array


lib = load_lib()
lib.c_roc_auc_score.restype = ctypes.c_double
_roc_auc_score = lib.c_roc_auc_score


def roc_auc_score(y_true, y_pred):
    y_true_ptr, _, _ = c_float_array(y_true)
    y_pred_ptr, _, _ = c_float_array(y_pred)
    num_data = y_true.shape[0]
    auc = _roc_auc_score(y_true_ptr, y_pred_ptr, num_data)
    return auc
