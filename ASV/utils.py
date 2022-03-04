import os
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

def init_args(args):
	args.score_save_path    = os.path.join(args.save_path, 'score.txt')
	args.model_save_path    = os.path.join(args.save_path, 'model')
	os.makedirs(args.model_save_path, exist_ok = True)
	return args

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_all_EERs(preds, keys):
    sasv_labels, sv_labels, spf_labels = [], [], []
    sv_preds, spf_preds = [], []

    for pred, key in zip(preds, keys):
        if key == "target":
            sasv_labels.append(1)
            sv_labels.append(1)
            spf_labels.append(1)
            sv_preds.append(pred)
            spf_preds.append(pred)

        elif key == "nontarget":
            sasv_labels.append(0)
            sv_labels.append(0)
            sv_preds.append(pred)

        elif key == "spoof":
            sasv_labels.append(0)
            spf_labels.append(0)
            spf_preds.append(pred)
        else:
            raise ValueError(
                f"should be one of 'target', 'nontarget', 'spoof', got:{key}"
            )

    fpr, tpr, _ = roc_curve(sasv_labels, preds, pos_label=1)    # 9717
    sasv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(sv_labels, sv_preds, pos_label=1)   # 328
    sv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(spf_labels, spf_preds, pos_label=1) # 9607
    spf_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return sasv_eer, sv_eer, spf_eer