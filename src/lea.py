import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionAnchorLossFunction(nn.Module):
    def __init__(self, alpha: float) -> None:
        super(EmotionAnchorLossFunction, self).__init__()
        self.alpha = alpha
    
    def compute_lea(self, emb_pos, emb_neu, emb_neg):
        batch_size = emb_pos.shape[0]
        fro_pos_neu =  torch.linalg.matrix_norm(emb_pos.T@emb_neu, ord='fro')
        fro_neg_neu =  torch.linalg.matrix_norm(emb_neg.T@emb_neu, ord='fro')
        dot_pos_neg = torch.sum(emb_pos * emb_neg, dim=-1).sum(dim=0)
        return (fro_pos_neu + fro_neg_neu + dot_pos_neg) / batch_size
        
    def forward(self, y_pred, y_true, emb_pos, emb_neu, emb_neg):
        l_cls = F.cross_entropy(input=y_pred, target=y_true)
        l_ea = self.compute_lea(emb_pos, emb_neu, emb_neg)
        return l_cls + self.alpha * l_ea