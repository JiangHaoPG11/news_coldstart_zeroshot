import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *

class discriminator(torch.nn.Module):
    def __init__(self, user_embedding_dim, item_embedding_dim, hidden_dim):
        super(discriminator, self).__init__()
        self.dropout_prob = 0.2
        self.mlp1 = nn.Linear(user_embedding_dim, hidden_dim, bias=True)
        self.mlp2 = nn.Linear(item_embedding_dim, hidden_dim, bias=True)
        self.tanh = nn.Tanh()
    def forward(self, user_embedding, item_embedding):
        user_embedding = F.dropout(user_embedding, p=self.dropout_prob, training=self.training)
        user_rep = self.tanh(self.mlp1(user_embedding))
        item_embedding = F.dropout(item_embedding, p=self.dropout_prob, training=self.training)
        item_rep = self.tanh(self.mlp2(item_embedding))
        out_dis = torch.sum(user_rep * item_rep, dim = -1)
        return out_dis

class generator(torch.nn.Module):
    def __init__(self, content_embedding_dim, hidden_dim):
        super(generator, self).__init__()
        self.mlp1 = nn.Linear(content_embedding_dim, hidden_dim, bias=True)
        self.dropout_prob = 0.2
        self.tanh = nn.Tanh()
    def forward(self, content_embedding):
        content_embedding = F.dropout(content_embedding, p=self.dropout_prob, training=self.training)
        content_rep = self.tanh(self.mlp1(content_embedding))
        return content_rep

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

        # 副主题级表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = torch.tanh(self.fc2(subcategory_embedding))

        # 单词级新闻表征
        # word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = self.multiheadatt(word_embedding)
        word_embedding = self.norm(word_embedding)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_rep = torch.tanh(self.word_attention(word_embedding))

        # 实体级新闻表征
        entity_embedding = torch.tanh(self.fc3(entity_embedding))
        entity_inter = self.GCN(entity_embedding)
        entity_inter = self.norm(entity_inter)
        entity_inter = F.dropout(entity_inter, p=self.dropout_prob, training=self.training)
        entity_rep = torch.tanh(self.entity_attention(entity_inter))

        # 新闻附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), entity_rep.unsqueeze(1),
                              category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = torch.tanh(self.news_attention(news_rep))

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
        return user_rep


class GAN_exp1(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(GAN_exp1, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding.to(device)
        self.relation_embedding = relation_embedding.to(device)

        # embedding
        self.category_embedding = nn.Embedding(self.args.category_num, self.args.embedding_dim).to(device)
        self.subcategory_embedding = nn.Embedding(self.args.subcategory_num, self.args.embedding_dim).to(device)

        # GAN_exp1
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.news_entity_size, self.args.entity_embedding_dim,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.category_num, self.args.subcategory_num)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.attention_dim,
                                         self.args.attention_heads, self.args.query_vector_dim,
                                         self.args.news_entity_size, self.args.entity_embedding_dim,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.category_num, self.args.subcategory_num)

        # build discriminator and generator
        self.discriminator = discriminator(self.args.attention_heads * self.args.attention_dim,
                                             self.args.attention_heads * self.args.attention_dim,
                                             self.args.embedding_dim)

        self.generator = generator(self.args.attention_heads * self.args.attention_dim,
                                   self.args.attention_heads * self.args.attention_dim)


        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict

    def get_news_entities(self, newsids):
        news_entities = []
        newsids = newsids.unsqueeze(-1)
        for i in range(newsids.shape[0]):
            news_entities.append([])
            for j in range(newsids.shape[1]):
                # news_entities[-1].append([])
                news_entities[-1].append(self.news_entity_dict[int(newsids[i, j])][:self.args.news_entity_size])
        return np.array(news_entities)
    
    def get_neighor_entities(self, entity, k=5):
        neighor_entity = []
        neighor_relation = []
        if len(entity.shape) == 2:
            for i in range(entity.shape[0]):
                neighor_entity.append([])
                neighor_relation.append([])
                for j in range(entity.shape[1]):
                    if entity[i, j] in self.entity_adj.keys():
                        neighor_entity[-1].append([])
                        neighor_entity[-1][-1].append(self.entity_adj[int(entity[i, j])][:k])
                        neighor_relation[-1].append([])
                        neighor_relation[-1][-1].append(self.relation_adj[int(entity[i, j])][:k])
                    else:
                        neighor_entity[-1].append([])
                        neighor_entity[-1][-1].append([0] * k)
                        neighor_relation[-1].append([])
                        neighor_relation[-1][-1].append([0] * k)
        elif len(entity.shape) == 3:
            for i in range(entity.shape[0]):
                neighor_entity.append([])
                neighor_relation.append([])
                for j in range(entity.shape[1]):
                    neighor_entity[-1].append([])
                    neighor_relation[-1].append([])
                    for m in range(entity.shape[2]):
                        if entity[i, j, m] in self.entity_adj.keys():
                            neighor_entity[-1][-1].append([])
                            neighor_entity[-1][-1][-1].append(self.entity_adj[int(entity[i, j, m])][:k])
                            neighor_relation[-1][-1].append([])
                            neighor_relation[-1][-1][-1].append(self.relation_adj[int(entity[i, j, m])][:k])
                        else:
                            neighor_entity[-1][-1].append([])
                            neighor_entity[-1][-1][-1].append([0] * k)
                            neighor_relation[-1][-1].append([])
                            neighor_relation[-1][-1][-1].append([0] * k)
        return np.array(neighor_entity), np.array(neighor_relation)

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        # 新闻单词
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        # 新闻实体
        candidate_news_entity_embedding = self.entity_embedding[self.get_news_entities(candidate_news_index)].to(self.device).squeeze()
        user_clicked_news_entity_embedding = self.entity_embedding[self.get_news_entities(user_clicked_news_index)].to(self.device).squeeze()
        # 新闻主题
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
            news_entity_embedding_one = candidate_news_entity_embedding[:, i, :, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one = self.news_encoder(news_word_embedding_one, news_entity_embedding_one,
                                             news_category_index, news_subcategory_index).unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            # 点击新闻单词嵌入
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # 点击新闻实体嵌入
            clicked_news_entity_embedding_one = user_clicked_news_entity_embedding[i, :, :, :]
            clicked_news_entity_embedding_one = clicked_news_entity_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_word_embedding_one, clicked_news_entity_embedding_one,
                                             clicked_news_category_index, clicked_news_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)

        news_content_rep = self.generator(news_rep)
        return user_rep, news_rep, news_content_rep

    def cal_d_loss(self, user_rep, news_rep, news_content_rep):
        real_out = self.discriminator(user_rep, news_rep)
        fake_out = self.discriminator(user_rep, news_content_rep)
        logit = real_out - fake_out
        d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(),  0 , 1),
                                                            torch.flatten(torch.ones_like(real_out), 0, 1))
        return d_loss

    def cal_g_loss(self, user_rep, news_rep, news_content_rep):
        g_out = self.discriminator(user_rep, news_content_rep)
        d_out = self.discriminator(user_rep, news_rep)
        logit = g_out - d_out
        g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(), 0 , 1),
                                                            torch.flatten(torch.ones_like(g_out), 0, 1))
                                                            
        sim_loss = 0.1 * torch.mean(torch.square(news_rep - news_content_rep))
        g_loss += sim_loss
        return g_loss

    def forward(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep, news_content_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        d_loss = self.cal_d_loss(user_rep, news_rep, news_content_rep)
        g_loss = self.cal_g_loss(user_rep, news_rep, news_content_rep)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score, d_loss, g_loss

    def test(self, candidate_news, user_clicked_news_index):
        # candidate_news = torch.flatten(candidate_news, 0, 1)
        # 新闻用户表征
        user_rep, news_rep, news_content_rep = self.get_user_news_rep(candidate_news, user_clicked_news_index)
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        return score
