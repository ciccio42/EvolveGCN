import torch
import utils as u
from pygod.nn.functional import double_recon_loss


class ReconstructionLoss(torch.nn.Module):
    """docstring for Cross_Entropy"""

    def __init__(self, args, dataset):
        super().__init__()
        self.weight = args.loss["weight"]
        self.pos_weight_a = args.loss["pos_weight_a"]
        self.pos_weight_s = args.loss["pos_weight_s"]
        self.bce_s = args.loss["bce_s"]

    def forward(self, pred_adj, gt_adj, pred_attri, gt_attri):

        loss = double_recon_loss(x=gt_attri.to_dense(),
                                 x_=pred_attri.to_dense(),
                                 s=gt_adj.to_dense(),
                                 s_=pred_adj.to_dense(),
                                 weight=self.weight,
                                 pos_weight_a=self.pos_weight_a,
                                 pos_weight_s=self.pos_weight_s,
                                 bce_s=self.bce_s)
        return loss
