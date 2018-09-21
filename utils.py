import torch

def  token_precision_recall(y_pred, y_true, unk_idx, pad_idx):
    """
    Get the precision/recall for the given token.
    :param predicted_parts: a list of predicted parts
    :param gold_set_parts: a list of the golden parts
    :return: precision, recall, f1 as floats
    """
    
    ground_truth = y_true[:]
    
    tp = 0
    for subtoken in set(y_pred):
        if subtoken == unk_idx or subtoken == pad_idx:
            continue
        if subtoken in ground_truth:
            ground_truth.remove(subtoken)
            tp += 1

    assert tp <= len(y_pred), (tp, len(y_pred))
    
    if len(y_pred) > 0:
        precision = float(tp) / len(y_pred)
    else:
        precision = 0

    assert tp <= len(y_true), (y_true)
    
    if len(y_true) > 0:
        recall = float(tp) / len(y_true)
    else:
        recall = 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.
        
    return precision, recall, f1

def logsumexp(x, y):
    max = torch.where(x > y, x, y)
    min = torch.where(x > y, y, x)
    return torch.log1p(torch.exp(min - max)) + max