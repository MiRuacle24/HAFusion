from parse_args import args
import numpy as np
import torch

def load_data():
    data_path = args.data_path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    landUse_feature = np.load(data_path + args.landUse_dist)
    landUse_feature = landUse_feature[np.newaxis]
    landUse_feature = torch.Tensor(landUse_feature).to(device)

    POI_feature = np.load(data_path + args.POI_dist)
    POI_feature = POI_feature[np.newaxis]
    POI_feature = torch.Tensor(POI_feature).to(device)

    mob_feature = np.load(data_path + args.mobility_dist)
    mob_feature = mob_feature[np.newaxis]
    mob_feature = torch.Tensor(mob_feature).to(device)

    mob_adj = np.load(data_path + args.mobility_adj)
    mob_adj = mob_adj/np.mean(mob_adj)
    mob_adj = torch.Tensor(mob_adj).to(device)

    poi_sim = np.load(data_path + args.POI_simi)
    poi_sim = torch.Tensor(poi_sim).to(device)

    land_sim = np.load(data_path + args.landUse_simi)
    land_sim = torch.Tensor(land_sim).to(device)

    features = [POI_feature, landUse_feature, mob_feature]

    return features, mob_adj, poi_sim, land_sim
