import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from parse_args import args

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class RegionFusionBlock(nn.Module):

    def __init__(self, input_dim, nhead, dropout, dim_feedforward=2048):
        super(RegionFusionBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward, )
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        src2, _ = self.self_attn(src, src, src, )

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class intraAFL_Block(nn.Module):

    def __init__(self, input_dim, nhead, c, dropout, dim_feedforward=2048):
        super(intraAFL_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(input_dim, nhead, dropout=dropout, batch_first=True, bias=True)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(input_dim, dim_feedforward, )
        self.linear2 = nn.Linear(dim_feedforward, input_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.expand = nn.Conv2d(1, c, kernel_size=1)
        self.pooling = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.proj = nn.Linear(c, input_dim)

        self.activation = F.relu

    def forward(self, src):
        src2, attnScore = self.self_attn(src, src, src, )
        attnScore = attnScore[:, np.newaxis]

        edge_emb = self.expand(attnScore)
        # edge_emb = self.pooling(edge_emb)
        w = edge_emb
        w = w.softmax(dim=-1)
        w = (w * edge_emb).sum(-1).transpose(-1, -2)
        w = self.proj(w)
        src2 = src2 + w

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class intraAFL(nn.Module):
    def __init__(self, input_dim, c):
        super(intraAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_IntraAFL
        NO_head = args.NO_head
        dropout = args.dropout

        self.blocks = nn.ModuleList(
            [intraAFL_Block(input_dim=input_dim, nhead=NO_head, c=c, dropout=dropout) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class RegionFusion(nn.Module):
    def __init__(self, input_dim):
        super(RegionFusion, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_RegionFusion
        NO_head = args.NO_head
        dropout = args.dropout

        self.blocks = nn.ModuleList(
            [RegionFusionBlock(input_dim=input_dim, nhead=NO_head, dropout=dropout) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class interAFL_Block(nn.Module):

    def __init__(self, d_model, S):
        super(interAFL_Block, self).__init__()
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)
        attn = self.softmax(attn)
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)

        return out


class interAFL(nn.Module):
    def __init__(self, input_dim, d_m):
        super(interAFL, self).__init__()
        self.input_dim = input_dim
        self.num_block = args.NO_InterAFL

        self.blocks = nn.ModuleList(
            [interAFL_Block(input_dim, d_m) for _ in range(self.num_block)])

        self.fc = DeepFc(input_dim, input_dim)

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        out = out.squeeze()
        out = self.fc(out)
        return out


class ViewFusion(nn.Module):
    def __init__(self, emb_dim, out_dim):
        super(ViewFusion, self).__init__()
        self.W = nn.Conv1d(emb_dim, out_dim, kernel_size=1, bias=False)
        self.f1 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.f2 = nn.Conv1d(out_dim, 1, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, src):
        seq_fts = self.W(src)
        f_1 = self.f1(seq_fts)
        f_2 = self.f2(seq_fts)
        logits = f_1 + f_2.transpose(1, 2)
        coefs = torch.mean(self.act(logits), dim=-1)
        coefs = torch.mean(coefs, dim=0)
        coefs = F.softmax(coefs, dim=-1)
        return coefs


class HAFusion(nn.Module):
    def __init__(self, poi_dim, landUse_dim, input_dim, output_dim, d_prime, d_m, c):
        super(HAFusion, self).__init__()
        self.input_dim = input_dim
        self.densePOI2 = nn.Linear(poi_dim, input_dim)
        self.denseLandUse3 = nn.Linear(landUse_dim, input_dim)

        self.encoderPOI = intraAFL(input_dim, c)
        self.encoderLandUse = intraAFL(input_dim, c)
        self.encoderMob = intraAFL(input_dim, c)

        self.regionFusionLayer = RegionFusion(input_dim)

        self.interViewEncoder = interAFL(input_dim, d_m)

        self.fc = DeepFc(input_dim, output_dim)

        self.para1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True) 
        self.para1.data.fill_(0.1)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True) 
        self.para2.data.fill_(0.9)

        self.viewFusionLayer = ViewFusion(input_dim, d_prime)

        self.activation = F.relu
        self.dropout = nn.Dropout(0.1)
        self.decoder_s = nn.Linear(output_dim, output_dim)  #
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.decoder_p = nn.Linear(output_dim, output_dim)  #
        self.decoder_l = nn.Linear(output_dim, output_dim)
        self.feature = None

    def forward(self, x):
        poi_emb, landUse_emb, mob_emb = x

        poi_emb = self.dropout(self.activation(self.densePOI2(poi_emb)))
        landUse_emb = self.dropout(self.activation(self.denseLandUse3(landUse_emb)))

        poi_emb = self.encoderPOI(poi_emb)
        landUse_emb = self.encoderLandUse(landUse_emb)
        mob_emb = self.encoderMob(mob_emb)

        out = torch.stack([poi_emb, landUse_emb, mob_emb])

        intra_view_embs = out
        out = out.transpose(0, 1)
        out = self.interViewEncoder(out)
        out = out.transpose(0, 1)
        p1 = self.para1 / (self.para1 + self.para2)
        p2 = self.para2 / (self.para1 + self.para2)
        out = out * p2 + intra_view_embs * p1
        # ---------------------------------------------

        out1 = out.transpose(0, 2)
        coef = self.viewFusionLayer(out1)
        temp_out = coef[0] * out[0] + coef[1] * out[1] + coef[2] * out[2]
        # --------------------------------------------------

        temp_out = temp_out[np.newaxis]
        temp_out = self.regionFusionLayer(temp_out)
        out = self.fc(temp_out)

        self.feature = out

        out_s = self.decoder_s(out)  # source embedding of regions
        out_t = self.decoder_t(out)  # destination embedding of regions
        out_p = self.decoder_p(out)  # poi embedding of regions
        out_l = self.decoder_l(out)  # landuse embedding of regions
        return out_s, out_t, out_p, out_l


    def out_feature(self):
        return self.feature