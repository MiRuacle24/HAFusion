import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import utils
from parse_args import args
import tasks_NY.tasks_crime, tasks_NY.tasks_chk, tasks_NY.tasks_serviceCall
import tasks_Chi.tasks_crime, tasks_Chi.tasks_chk, tasks_Chi.tasks_serviceCall
import tasks_SF.tasks_crime, tasks_SF.tasks_chk, tasks_SF.tasks_serviceCall
from HAFusion_Model import HAFusion

features, mob_adj, poi_sim, land_sim = utils.load_data()

city = args.city
embedding_size = args.embedding_size
d_prime = args.d_prime
d_m = args.d_m
c = args.c 
POI_dim = args.POI_dim
landUse_dim = args.landUse_dim
region_num = args.region_num
task = args.task

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

    def forward(self, out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim):
        mob_loss = _mob_loss(out_s, out_t, mob_adj)
        poi_loss = _general_loss(out_p, poi_sim)
        land_loss = _general_loss(out_l, land_sim)
        loss = poi_loss + land_loss + mob_loss
        return loss
    
def train_model(input_features, mob_adj, poi_sim, land_sim, model, model_loss, city, task):
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_emb = 0
    best_r2 = 0

    for epoch in range(epochs):
        model.train()
        out_s, out_t, out_p, out_l = model(input_features)
        loss = model_loss(out_s, out_t, mob_adj, out_p, poi_sim, out_l, land_sim)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 30 == 0:
            print("Epoch {}, Loss {}".format(epoch, loss.item()))
            embs = model.out_feature()
            embs = embs.detach().cpu().numpy()

            if task == "checkIn":
                if city == "NY":
                    _, _, r2 = tasks_NY.tasks_chk.do_tasks(embs)
                elif city == "Chi":
                    _, _, r2 = tasks_Chi.tasks_chk.do_tasks(embs)
                elif city == "SF":
                    _, _, r2 = tasks_SF.tasks_chk.do_tasks(embs)
            elif task == "crime":
                if city == "NY":
                    _, _, r2 = tasks_NY.tasks_crime.do_tasks(embs)
                elif city == "Chi":
                    _, _, r2 = tasks_Chi.tasks_crime.do_tasks(embs)
                elif city == "SF":
                    _, _, r2 = tasks_SF.tasks_crime.do_tasks(embs)
            elif task == "serviceCall":
                if city == "NY":
                    _, _, r2 = tasks_NY.tasks_serviceCall.do_tasks(embs)
                elif city == "Chi":
                    _, _, r2 = tasks_Chi.tasks_serviceCall.do_tasks(embs)
                elif city == "SF":
                    _, _, r2 = tasks_SF.tasks_serviceCall.do_tasks(embs)

            if best_r2 < r2:
                best_r2 = r2
                best_emb = embs

    np.save("best_emb.npy", best_emb)

def test_model(city, task):
    best_emb = np.load("./best_emb.npy")
    print("Best region embeddings")
    if task == "checkIn":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Check-In in New York City')
            mae, rmse, r2 = tasks_NY.tasks_chk.do_tasks(best_emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Check-In in Chicago')
            mae, rmse, r2 = tasks_Chi.tasks_chk.do_tasks(best_emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Check-In in San Francisco')
            mae, rmse, r2 = tasks_SF.tasks_chk.do_tasks(best_emb)
    elif task == "crime":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Crime in New York City')
            mae, rmse, r2 = tasks_NY.tasks_crime.do_tasks(best_emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Crime in Chicago')
            mae, rmse, r2 = tasks_Chi.tasks_crime.do_tasks(best_emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Crime in San Francisco')
            mae, rmse, r2 = tasks_SF.tasks_crime.do_tasks(best_emb)
    elif task == "serviceCall":
        if city == "NY":
            print('>>>>>>>>>>>>>>>>>   Service Calls in New York City')
            mae, rmse, r2 = tasks_NY.tasks_serviceCall.do_tasks(best_emb)
        elif city == "Chi":
            print('>>>>>>>>>>>>>>>>>   Service Calls in Chicago')
            mae, rmse, r2 = tasks_Chi.tasks_serviceCall.do_tasks(best_emb)
        elif city == "SF":
            print('>>>>>>>>>>>>>>>>>   Service Calls in San Francisco')
            mae, rmse, r2 = tasks_SF.tasks_serviceCall.do_tasks(best_emb)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HAFusion(POI_dim, landUse_dim, region_num, embedding_size, d_prime, d_m, c).to(device)
    model_loss = ModelLoss()
    
    print('Model Training-----------------')
    model.train()
    train_model(features, mob_adj, poi_sim, land_sim, model, model_loss, city, task)

    print("Downstream task test-----------")
    test_model(city, task)


