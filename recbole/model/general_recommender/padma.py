import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym import optim

from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss
from recbole.utils import InputType


class PADMA(GeneralRecommender):

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(PADMA, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.MLPHiddenSize = config["mlp_hidden_size"]
        self.mae_hidden_size = config["mae_hidden_size"]

        self.mask_ratio = config["mask_ratio"]
        self.gamma = config["gamma"]
        self.inner_loop = config["inner_loop"]
        self.inner_lr = config["inner_lr"]
        self.alpha = config["alpha"]
        self.cf_aug_cnt = config["cf_aug_cnt"]

        # load dataset info
        self.mask_token = self.n_items
        # self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        self.droupout_rate = 0.1

        self.llm_aug = config['llm_aug']
        self.aug_user_ids = torch.tensor([x for x in list(self.llm_aug.keys())], device='cuda')
        self.aug_pos_items = torch.tensor([v['1'] for v in self.llm_aug.values()], device='cuda')
        self.aug_neg_items = torch.tensor([v['2'] for v in self.llm_aug.values()], device='cuda')

        # define layers and loss
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size)# add mask_token at the last

        self.meta_mae = MetaMAE(self.MLPHiddenSize, self.mae_hidden_size)

        self.loss = BPRLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

        self.optimizer = optim.Adam(self.meta_mae.parameters(), lr=self.inner_lr)

    def get_user_embedding(self, user):
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)
        return user_e, item_e

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        # Splicing augmentation samples behind sampled interactions
        unique_user_ids = torch.unique(user)
        shuffled_indices = torch.randperm(unique_user_ids.size(0))
        unique_user_ids = unique_user_ids[shuffled_indices][:int(0.2 * unique_user_ids.size(0))]
        # 查找unique_user_ids中每个用户ID在user_ids中的位置
        indices = torch.nonzero(unique_user_ids[:, None] == self.aug_user_ids, as_tuple=True)[1]

        aug_uid = unique_user_ids
        aug_pos_iid = self.aug_pos_items[indices]
        aug_neg_iid = self.aug_neg_items[indices]

        org_user = user.clone()
        org_pos_item = pos_item.clone()
        org_neg_item = neg_item.clone()
        user = torch.cat((user, aug_uid))
        pos_item = torch.cat((pos_item, aug_pos_iid))
        neg_item = torch.cat((neg_item, aug_neg_iid))

        # mask
        mask = torch.rand(aug_pos_iid.shape) < self.mask_ratio
        masked_aug_pos_iid = aug_pos_iid.clone()
        masked_aug_pos_iid[mask] = self.mask_token

        user_e, pos_e = self.forward(user, pos_item)

        for i in self.inner_loop:
            self.optimizer.zero_grad()
            h_hat, z = self.meta_mae(pos_e)

            # Compute norms of the vectors
            norm_H_hat_u = torch.norm(h_hat)
            norm_H_u = torch.norm(pos_e)
            # Initialize loss value
            loss = 0.0
            # Compute the dot product between H_hat_u and H_u
            dot_product = np.dot(h_hat, pos_e)
            # Compute the term inside the summation
            term = 1 - (dot_product / (norm_H_hat_u * norm_H_u))
            # Raise the term to the power of gamma and add it to the loss
            loss += term ** self.gamma
            # Take the average by dividing by the number of elements in V_u_t
            L_fr = loss / len(aug_uid)
            L_fr_tensor = torch.tensor(L_fr, requires_grad=True)
            L_fr_tensor.backward()

            # Update parameters
            self.optimizer.step()

        h_hat, z = self.meta_mae(pos_e)
        # get the final embedding of user
        for i, user_id in enumerate(indices):
            user_id = user_id.item()  # Get the user ID as an integer
            user_e[user_id] = (1 - self.alpha) * user_e[user_id] + self.alpha * z[i]

        score = self.full_sort_predict(interaction)

        _, top_3_item_ids = torch.topk(score, self.cf_aug_cnt, dim=1)
        # Get user IDs along with top 3 item IDs
        user_ids = torch.arange(2048).unsqueeze(1).repeat(1, 3)
        user_top_3_item_ids = torch.stack((user_ids, top_3_item_ids), dim=2)

        user = torch.cat((user, user_top_3_item_ids))
        pos_item = torch.cat((pos_item, top_3_item_ids))

        user_e, pos_e = self.forward(user, pos_item)
        neg_e = self.get_item_embedding(neg_item)
        pos_item_score, neg_item_score = torch.mul(user_e, pos_e).sum(dim=1), torch.mul(
            user_e, neg_e
        ).sum(dim=1)
        loss_aug = self.loss(pos_item_score, neg_item_score)

        org_user_e, org_pos_e = self.forward(org_user, org_pos_item)
        org_neg_e = self.get_item_embedding(org_neg_item)
        org_pos_item_score, org_neg_item_score = torch.mul(org_user_e, org_pos_e).sum(dim=1), torch.mul(
            org_user_e, org_neg_e
        ).sum(dim=1)
        loss = self.loss(org_pos_item_score, org_neg_item_score)

        return loss, loss_aug

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)


class MetaMAE(nn.Module):
    def __init__(self, MLPHiddenSize, mae_hidden_size):
        super(MetaMAE, self).__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            MLPLayers(MLPHiddenSize),
            nn.Linear(MLPHiddenSize[-1], mae_hidden_size)
        )
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(mae_hidden_size, MLPHiddenSize[-1]),
            MLPLayers(list(reversed(MLPHiddenSize)))
        )

    def forward(self, x):
        # Pass through encoder, then decoder
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

