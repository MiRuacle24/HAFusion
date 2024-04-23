import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from parse_args import args

def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = torch.mm(s_embeddings, t_embeddings.T)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat + 0.0001)))
    inner_prod = torch.mm(t_embeddings, s_embeddings.T)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat + 0.0001)))
    return loss


def _general_loss(embeddings, adj):
    inner_prod = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=2)
    loss = F.mse_loss(inner_prod, adj)
    return loss


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()

    def forward(self, out_s, out_t, mob_adj):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        loss =  mob_loss
        return loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mob_adj = np.load(args.data_path + args.mobility_adj)
mob_adj = mob_adj/np.mean(mob_adj)
mob_adj = torch.Tensor(mob_adj).to(device)

emb = np.load("best_emb.npy")
emb = torch.Tensor(emb).to(device)
a = _mob_loss(emb, emb, mob_adj)


mob_adj1 = np.load("./data_NY/mob-adj.npy")
mob_adj1 = mob_adj1[np.newaxis]
mob_adj1 = mob_adj1/np.mean(mob_adj1, axis=(1, 2))
mob_adj1 = mob_adj1[0]
mob_adj1 = torch.Tensor(mob_adj1).to(device)
b = _mob_loss(emb, emb, mob_adj1)

print(a == b)