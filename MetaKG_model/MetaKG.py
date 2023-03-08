import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *
import scipy.sparse as sp
from collections import OrderedDict
import random

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.multi_dim = attention_dim * attention_heads
        self.norm = nn.LayerNorm(self.multi_dim)

        # 主题级表征网络
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.fc1 = nn.Linear(category_dim, self.multi_dim, bias=True)

        # 副主题级表征网络
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc2 = nn.Linear(subcategory_dim, self.multi_dim, bias=True)

        # 单词级表征网络
        self.multiheadatt = MultiHeadSelfAttention_2(word_dim, attention_dim * attention_heads, attention_heads)
        self.word_attention = Additive_Attention(query_vector_dim, self.multi_dim)

        # 实体级表征网络
        self.fc3 = nn.Linear(entity_embedding_dim, entity_embedding_dim, bias=True)
        self.GCN = gcn(entity_size, entity_embedding_dim, self.multi_dim)
        self.entity_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.news_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 主题级表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = torch.tanh(self.fc1(category_embedding))
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = torch.tanh(self.fc2(subcategory_embedding))
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # 单词级新闻表征
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = torch.tanh(self.word_attention(word_embedding))
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)

        # 实体级新闻表征
        entity_embedding = torch.tanh(self.fc3(entity_embedding))
        entity_inter = self.GCN(entity_embedding)
        entity_inter = self.norm(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = torch.tanh(self.entity_attention(entity_inter))
        entity_rep = F.dropout(entity_rep, p=self.dropout_prob, training=self.training)

        # 新闻附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1),
                              category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = torch.tanh(self.news_attention(news_rep))
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)

        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, attention_dim, attention_heads, query_vector_dim, entity_size, entity_embedding_dim,
                 category_dim, subcategory_dim, category_size, subcategory_size):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(word_dim, attention_dim, attention_heads, query_vector_dim, entity_size,
                                         entity_embedding_dim, category_dim, subcategory_dim, category_size, subcategory_size)
        self.multi_dim = attention_dim * attention_heads
        self.multiheadatt = MultiHeadSelfAttention_2(self.multi_dim, self.multi_dim, attention_heads)

        self.norm = nn.LayerNorm(self.multi_dim)

        self.user_attention = Additive_Attention(query_vector_dim, self.multi_dim)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, entity_embedding, category_index, subcategory_index):
        # 点击新闻表征
        news_rep = self.news_encoder(word_embedding, entity_embedding,
                                     category_index, subcategory_index).unsqueeze(0)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        news_rep = self.multiheadatt(news_rep)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        # 用户表征
        user_rep = torch.tanh(self.user_attention(news_rep))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)

        return user_rep

class Aggregator(nn.Module):
    def __init__(self, device, news_entity_dict, entity_adj, relation_adj):
        super(Aggregator, self).__init__()
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.device = device
        self.news_attention = nn.Linear(100, 1, bias=True)
        self.entity_attention = nn.Linear(100, 1, bias=True)

    def get_news_entities_batch(self):
        news_entities = []
        news_relations = []
        news = []
        for key, value in self.news_entity_dict.items():
            news.append(key)
            news_entities.append(value)
            news_relations.append([0 for k in range(len(value))])
        news = torch.tensor(news).to(self.device)
        news_entities = torch.tensor(news_entities).to(self.device)
        news_relations = torch.tensor(news_relations).to(self.device)# bz, news_entity_num
        return news, news_entities, news_relations

    def get_entities_neigh_batch(self, n_entity):
        neigh_entities = []
        neigh_relations = []
        entities = []
        for i in range(n_entity):
            if i in self.entity_adj.keys():
                entities.append(i)
                neigh_entities.append(self.entity_adj[i])
                neigh_relations.append(self.relation_adj[i])
            else:
                entities.append(i)
                neigh_entities.append([0 for k in range(20)])
                neigh_relations.append([0 for k in range(20)])
        entities = torch.tensor(entities).to(self.device)
        neigh_entities = torch.tensor(neigh_entities).to(self.device)
        neigh_relations = torch.tensor(neigh_relations).to(self.device) # bz, news_entity_num
        return entities, neigh_entities, neigh_relations

    def forward(self, user_emb, node_embedding, entity_emb, relation_emb, interact_mat):

        newsid, news_entities, news_relations = self.get_news_entities_batch()
        news_emb = node_embedding[newsid]
        news_neigh_entities_embedding = entity_emb[news_entities]
        news_neigh_relation_embedding = relation_emb[news_relations]
        news_weight = F.softmax(torch.tanh(self.news_attention(news_neigh_entities_embedding + news_neigh_relation_embedding)), dim = -1)
        news_agg = torch.matmul(torch.transpose(news_weight, -1, -2), news_neigh_entities_embedding).squeeze()

        entities, neigh_entities, neigh_relations = self.get_entities_neigh_batch(n_entity = len(entity_emb))
        entity_emb = node_embedding[entities]
        neigh_entities_embedding = entity_emb[neigh_entities]
        neigh_relation_embedding = relation_emb[neigh_relations]
        entity_weight = F.softmax(torch.tanh(self.entity_attention(neigh_relation_embedding + neigh_entities_embedding)), dim = -1)
        entity_agg = torch.matmul(torch.transpose(entity_weight, -1, -2), neigh_entities_embedding).squeeze()

        node_emb = torch.cat([news_agg + news_emb, entity_agg + entity_emb])
        user_agg = torch.sparse.mm(interact_mat, node_emb)
        user_agg = user_emb + user_agg  # [n_users, channel]
        return node_emb, user_agg

class metakg(nn.Module):
    def __init__(self, device, n_hops,
                 interact_mat, news_entity_dict,
                 entity_adj, relation_adj,
                 mess_dropout_rate=0.1):
        super(metakg, self).__init__()
        self.convs = nn.ModuleList()
        self.interact_mat = interact_mat
        self.mess_dropout_rate = mess_dropout_rate
        self.device = device

        for i in range(n_hops):
            self.convs.append(Aggregator(self.device, news_entity_dict,
                                         entity_adj, relation_adj))
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def forward(self, user_embedding, node_embedding, entity_embedding, relation_embedding, mess_dropout=True):

        node_res_emb = node_embedding
        user_res_emb = user_embedding

        for i in range(len(self.convs)):
            node_emb, user_emb = self.convs[i](user_embedding, node_embedding,
                                               entity_embedding, relation_embedding,
                                               self.interact_mat)
            if mess_dropout:
                node_emb = self.dropout(node_emb)
                user_emb = self.dropout(user_emb)
            node_emb = F.normalize(node_emb)
            user_emb = F.normalize(user_emb)
            node_res_emb = torch.add(node_res_emb, node_emb)
            user_res_emb = torch.add(user_res_emb, user_emb)
        return user_res_emb, node_res_emb


class MetaKG(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, user_click_dict, device):
        super(MetaKG, self).__init__()
        self.args = args
        self.device = device

        # # no_embedding
        # self.word_embedding = word_embedding
        # self.entity_embedding = entity_embedding.to(device)
        # self.relation_embedding = relation_embedding.to(device)
        #
        # # embedding
        # self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        # self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)
        #
        # # MetaKG
        # self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.attention_dim,
        #                                  self.args.attention_heads, self.args.query_vector_dim,
        #                                  self.args.news_entity_size, self.args.entity_embedding_dim,
        #                                  self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
        #                                  self.args.category_num, self.args.subcategory_num)
        # self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim,
        #                                  self.args.attention_heads, self.args.query_vector_dim,
        #                                  self.args.news_entity_size, self.args.entity_embedding_dim,
        #                                  self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
        #                                  self.args.category_num, self.args.subcategory_num)
        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

        # Meta
        self.n_users = self.args.user_num
        self.n_news = self.args.news_num
        self.n_entities = len(entity_embedding)
        self.n_relations = len(relation_embedding)
        
        # 嵌入
        self.user_embedding = nn.Embedding(self.n_users, self.args.embedding_dim)
        self.news_embedding = nn.Embedding(self.n_news, self.args.embedding_dim)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding)
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding)
        self.news_entity_dict = news_entity_dict
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.user_click_dict = user_click_dict
        
        # 交互矩阵
        self.interact_mat = self._convert_sp_mat_to_sp_tensor(self.user_click_dict, self.news_entity_dict).to(self.device)

        # 元学习参数
        self.num_inner_update = args.num_inner_update
        self.meta_update_lr = args.meta_update_lr
        self.decay = self.args.l2
        self.n_hops = self.args.n_hops
        self.mess_dropout = self.args.mess_dropout
        self.mess_dropout_rate = self.args.mess_dropout_rate
        
        # KGAT网络参数
        self.kgat = self._init_model()


    def _init_model(self):
        return metakg(device=self.device,
                      n_hops=self.n_hops,
                      interact_mat=self.interact_mat,
                      news_entity_dict=self.news_entity_dict,
                      entity_adj=self.entity_adj,
                      relation_adj=self.relation_adj,
                      mess_dropout_rate=self.mess_dropout_rate)

    def _get_parameter(self):
        param_dict = dict()
        for name, para in self.kgat.named_parameters():
            if name.startswith('conv'):
                param_dict[name] = para
        param_dict = OrderedDict(param_dict)
        return param_dict

        # update
    def _convert_sp_mat_to_sp_tensor(self, user_click_dict, news_entity_dict):
        adj = np.zeros([self.n_users, self.n_news + self.n_entities])
        for i in range(len(user_click_dict)):
            news_index = user_click_dict[i]
            for j in news_index:
                if j != self.n_news - 1:
                    adj[i][j] = 1
                    entity_list = news_entity_dict[i]
                    for m in entity_list:
                        adj[i][m + self.n_news] = 1
        X = sp.csr_matrix(adj, dtype=np.float32)
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def _concat_node_embedding(self):
        user_embeddings = self.user_embedding.weight
        news_embeddings = self.news_embedding.weight
        entity_embeddings = self.entity_embedding.weight
        relation_embeddings = self.relation_embedding.weight
        node_embedding = torch.cat([news_embeddings, entity_embeddings], dim=0)
        return user_embeddings, node_embedding, entity_embeddings, relation_embeddings

    def forward_meta(self, support_user_index, support_candidate_newsindex, support_labels,
                     query_user_index, query_candidate_newsindex, query_labels, fast_weights=None):
        support_user_index = support_user_index.to(self.device)
        support_candidate_newsindex = support_candidate_newsindex.to(self.device)
        support_labels = support_labels.to(self.device)

        query_user_index = query_user_index.to(self.device)
        query_candidate_newsindex = query_candidate_newsindex.to(self.device)
        query_labels = query_labels.to(self.device)

        user_embeddings, node_embedding, entity_embeddings, relation_embeddings = self._concat_node_embedding()

        if fast_weights == None:
            fast_weights = self._get_parameter()

        for i in range(self.num_inner_update):
            user_kgat_emb, node_kgat_emb = self.kgat(user_embeddings, node_embedding,
                                                     entity_embeddings, relation_embeddings,
                                                     mess_dropout=self.mess_dropout)
            u_s = user_kgat_emb[support_user_index]
            i_s = node_kgat_emb[support_candidate_newsindex]
            loss, _, _, _ = self.create_loss(u_s, i_s, support_labels)
            gradients = torch.autograd.grad(torch.mean(loss), self.kgat.parameters(), create_graph=False)
            fast_weights = OrderedDict(
                (name, param - self.meta_update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        user_kgat_emb, node_kgat_emb = self.kgat(user_embeddings, node_embedding,
                                                 entity_embeddings, relation_embeddings,
                                                 mess_dropout=self.mess_dropout)
        u_q = user_kgat_emb[query_user_index]
        i_q = node_kgat_emb[query_candidate_newsindex]
        return self.create_loss(u_q, i_q, query_labels)


    def forward(self, user_index, candidate_newsindex, labels):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_index = user_index.to(self.device)

        user_embeddings, node_embedding, entity_embeddings, relation_embeddings = self._concat_node_embedding()

        user_kgat_emb, node_kgat_emb = self.kgat(user_embeddings, node_embedding,
                                                 entity_embeddings, relation_embeddings,
                                                 mess_dropout=self.mess_dropout)
        u_e = user_kgat_emb[user_index]
        i_e = node_kgat_emb[candidate_newsindex]
        return self.create_loss(u_e, i_e, labels)

    def create_bpr_loss(self, user_embeddings, news_embeddings, labels):
        scores = (user_embeddings * news_embeddings).sum(dim=1)
        base_loss = torch.mean(nn.BCEWithLogitsLoss()(scores, labels))
        l2_loss = torch.norm(user_embeddings) ** 2 / 2 + torch.norm(news_embeddings) ** 2 / 2
        return base_loss + l2_loss, scores

    def create_loss(self, user_embeddings, news_embeddings, labels):
        batch_size = user_embeddings.shape[0]
        scores = (news_embeddings * user_embeddings.unsqueeze(1)).sum(dim=-1)
        rec_loss = nn.CrossEntropyLoss(reduce=False)(F.softmax(scores, dim = -1), torch.argmax(labels.to(self.device), dim=1))
        regularizer = (torch.norm(user_embeddings) ** 2
                       + torch.norm(news_embeddings) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        return rec_loss + emb_loss, scores, rec_loss, emb_loss

    def test(self, user_index, candidate_newsindex):
        candidate_newsindex = candidate_newsindex.to(self.device)
        user_index = user_index.to(self.device)

        user_embeddings, node_embedding, entity_embeddings, relation_embeddings = self._concat_node_embedding()
        user_kgat_emb, node_kgat_emb = self.kgat(user_embeddings, node_embedding,
                                                 entity_embeddings, relation_embeddings,
                                                 mess_dropout=self.mess_dropout)
        u_e = user_kgat_emb[user_index]
        i_e = node_kgat_emb[candidate_newsindex]

        scores = (i_e * u_e.unsqueeze(1)).sum(dim=-1)
        scores = torch.sigmoid(scores)
        return scores

    # def get_news_entities_batch(self, newsids):
    #     news_entities = []
    #     newsids = newsids.unsqueeze(-1)
    #     for i in range(newsids.shape[0]):
    #         news_entities.append([])
    #         for j in range(newsids.shape[1]):
    #             news_entities[-1].append([])
    #             news_entities[-1][-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.news_entity_size])
    #     return np.array(news_entities)
    #
    # def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
    #     # 新闻单词
    #     candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
    #     user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
    #     # 新闻实体
    #     candidate_news_entity_embedding = self.entity_embedding[self.get_news_entities_batch(candidate_news_index)].to(self.device).squeeze()
    #     user_clicked_news_entity_embedding = self.entity_embedding[self.get_news_entities_batch(user_clicked_news_index)].to(self.device).squeeze()
    #     # 新闻主题
    #     candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
    #     user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
    #     # 新闻副主题
    #     candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
    #     user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
    #
    #
    #     # 新闻编码器
    #     news_rep = None
    #     for i in range(self.args.sample_size):
    #         news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
    #         news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
    #         news_category_index = candidate_news_category_index[:, i]
    #         news_subcategory_index = candidate_news_subcategory_index[:, i]
    #         news_rep_one = self.news_encoder(news_word_embedding_one, news_entity_embedding_one,
    #                                          news_category_index, news_subcategory_index).unsqueeze(1)
    #         if i == 0:
    #             news_rep = news_rep_one
    #         else:
    #             news_rep = torch.cat([news_rep, news_rep_one], dim=1)
    #     # 用户编码器
    #     user_rep = None
    #     for i in range(self.args.batch_size):
    #         # 点击新闻单词嵌入
    #         clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
    #         clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
    #         # 点击新闻实体嵌入
    #         clicked_news_entity_embedding_one = user_clicked_news_entity_embedding[i, :, :, :]
    #         clicked_news_entity_embedding_one = clicked_news_entity_embedding_one.squeeze()
    #         # 点击新闻主题index
    #         clicked_news_category_index = user_clicked_news_category_index[i, :]
    #         # 点击新闻副主题index
    #         clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
    #         # 用户表征
    #         user_rep_one = self.user_encoder(clicked_news_word_embedding_one, clicked_news_entity_embedding_one,
    #                                          clicked_news_category_index, clicked_news_subcategory_index).unsqueeze(0)
    #         if i == 0:
    #             user_rep = user_rep_one
    #         else:
    #             user_rep = torch.cat([user_rep, user_rep_one], dim=0)
    #     return user_rep, news_rep

    # def forward(self, candidate_news, user_clicked_news_index):
    #     # candidate_news = torch.flatten(candidate_news, 0, 1)
    #     # 新闻用户表征
    #     user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
    #     # 预测得分
    #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
    #     return score
    #
    # def test(self, candidate_news, user_clicked_news_index):
    #     # candidate_news = torch.flatten(candidate_news, 0, 1)
    #     # 新闻用户表征
    #     user_rep, news_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
    #     # 预测得分
    #     score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
    #     return score
