import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *


class user_att_encoder(torch.nn.Module):
    def __init__(self, args):
        super(user_att_encoder, self).__init__()
        self.args = args
        self.category_dim = self.args.category_embedding_dim
        self.category_num = self.args.category_num
        self.multi_dim  = self.args.attention_dim * self.args.attention_heads
        self.category_embedding = nn.Embedding(self.category_num, self.category_dim)
        self.fc1 = nn.Linear(self.category_dim, self.multi_dim)
        self.attention = Additive_Attention(self.args.query_vector_dim, self.multi_dim)
        self.dropout_prob = 0.2
    
    def forward(self, user_interest_index):
        user_interest_embedding = self.category_embedding(user_interest_index)
        user_interest_embedding = torch.tanh(self.fc1(user_interest_embedding))
        user_interest_embedding = F.dropout(user_interest_embedding, p=self.dropout_prob, training=self.training)
        user_att_rep = torch.tanh(self.attention(user_interest_embedding))
        user_att_rep = F.dropout(user_att_rep, p=self.dropout_prob, training=self.training)
        return user_att_rep

class discriminator(torch.nn.Module):
    def __init__(self, user_embedding_dim, item_embedding_dim, hidden_dim):
        super(discriminator, self).__init__()
        self.dropout_prob = 0.2
        self.mlp1 = nn.Linear(user_embedding_dim, hidden_dim, bias=True)
        self.mlp2 = nn.Linear(item_embedding_dim, hidden_dim, bias=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, user_embedding, item_embedding):
        user_embedding = F.dropout(user_embedding, p=self.dropout_prob, training=self.training)
        user_rep = self.relu(self.bn1(self.mlp1(user_embedding)))
        item_embedding = F.dropout(item_embedding, p=self.dropout_prob, training=self.training)
        item_rep = self.relu(self.bn2(self.mlp2(item_embedding)))
        out_dis = torch.sum(user_rep * item_rep, dim = -1)
        return out_dis

class news_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size):
        super(news_encoder, self).__init__()
        self.embedding_layer1 = nn.Embedding(category_size, embedding_dim=category_dim)
        self.embedding_layer2 = nn.Embedding(subcategory_size, embedding_dim=subcategory_dim)
        self.fc1 = nn.Linear(category_dim, num_filters, bias=True)
        self.fc2 = nn.Linear(subcategory_dim, num_filters, bias=True)
        self.norm1 = nn.LayerNorm(num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.cnn = cnn(title_word_size, word_dim, dropout_prob, query_vector_dim, num_filters, window_sizes)
        self.norm3 = nn.LayerNorm(num_filters)
        self.news_attention = Additive_Attention(query_vector_dim, num_filters)
        self.norm4 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        # 主题表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        #category_rep = self.norm1(category_rep)
        # 副主题表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        #subcategory_rep = self.norm2(subcategory_rep)
        # 单词表征
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        #word_rep = self.norm3(word_rep)
        # 附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = torch.tanh(self.news_attention(news_rep))
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        # news_rep = self.norm4(news_rep)
        return news_rep, word_rep, category_rep, subcategory_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, title_word_size, category_dim, subcategory_dim,
                 dropout_prob, query_vector_dim, num_filters, window_sizes, category_size,subcategory_size):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(word_dim, title_word_size, category_dim, subcategory_dim,
                                       dropout_prob, query_vector_dim, num_filters, window_sizes, category_size, subcategory_size)
        self.user_attention = Additive_Attention(query_vector_dim, num_filters)
        self.norm2 = nn.LayerNorm(num_filters)
        self.dropout_prob = 0.2

    def forward(self, word_embedding, category_index, subcategory_index):
        news_rep, _, _, _ = self.news_encoder(word_embedding, category_index, subcategory_index)
        user_rep = torch.tanh(self.user_attention(news_rep.unsqueeze(0)))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        #user_rep = self.norm2(user_rep)
        return user_rep

class Zeroshot_GAN_NAML(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(Zeroshot_GAN_NAML, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        # encoder
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                        self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                        self.args.drop_prob, self.args.query_vector_dim,
                                        self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                        self.args.category_num, self.args.subcategory_num)

        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.title_word_size,
                                         self.args.category_embedding_dim, self.args.subcategory_embedding_dim,
                                         self.args.drop_prob, self.args.query_vector_dim,
                                         self.args.cnn_num_filters, self.args.cnn_window_sizes,
                                         self.args.category_num, self.args.subcategory_num)
        # build discriminator and generator
        self.discriminator_news = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)
        self.discriminator_user = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)


        # 用户和新闻ID嵌入
        # self.newsId_encoder = newsId_encoder(args)
        # self.userId_encoder = userId_encoder(args)
        
        # 用户和新闻属性嵌入
        self.userAtt_encoder = user_att_encoder(args)

        # zeroshot学习
        self.zeroshot_news_tower = zeroshot_news_simple_tower(args)
        self.zeroshot_user_tower = zeroshot_user_simple_tower(args) 

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
        # 新闻主题
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        news_word_rep = None
        news_category_rep = None
        news_subcategory_rep = None
        
        for i in range(self.args.sample_size):
            news_word_embedding_one = candidate_news_word_embedding[:, i, :, :]
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one, word_rep_one, category_rep_one, \
            subcategory_rep_one = self.news_encoder(news_word_embedding_one, news_category_index, news_subcategory_index)
            if i == 0:
                news_rep = news_rep_one.unsqueeze(1)
                news_word_rep = word_rep_one.unsqueeze(1)
                news_category_rep = category_rep_one.unsqueeze(1)
                news_subcategory_rep = subcategory_rep_one.unsqueeze(1)
            else:
                news_rep = torch.cat([news_rep, news_rep_one.unsqueeze(1)], dim=1)
                news_word_rep = torch.cat([news_word_rep, word_rep_one.unsqueeze(1)], dim=1)
                news_category_rep = torch.cat([news_category_rep, category_rep_one.unsqueeze(1)], dim=1)
                news_subcategory_rep = torch.cat([news_subcategory_rep, subcategory_rep_one.unsqueeze(1)], dim=1)

        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            # 点击新闻单词嵌入
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # 用户表征
            user_rep_one = self.user_encoder(clicked_news_word_embedding_one,
                                             clicked_news_category_index, clicked_news_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep, [news_word_rep, news_category_rep, news_subcategory_rep]

    def get_user_att_rep(self, user_clicked_news_index):
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        user_att_rep = self.userAtt_encoder(user_clicked_news_category_index)
        return user_att_rep

    def cal_news_d_loss(self, user_rep, news_rep, newsId_rep):
        real_out = self.discriminator_news(user_rep, news_rep)
        fake_out = self.discriminator_news(user_rep, newsId_rep)
        logit = real_out - fake_out
        # d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(),  0 , 1),
        #                                                     torch.flatten(torch.ones_like(real_out), 0, 1))
        d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(real_out))
        return d_loss

    def cal_news_g_loss(self, user_rep, news_rep, newsId_rep):
        g_out = self.discriminator_news(user_rep, newsId_rep)
        d_out = self.discriminator_news(user_rep, news_rep)
        logit = g_out - d_out
        # g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(), 0 , 1),
        #                                                     torch.flatten(torch.ones_like(g_out), 0, 1))
        g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(g_out))
        sim_loss = 0.1 * torch.mean(torch.square(news_rep - newsId_rep))
        # sim_loss = 0.1 * (torch.mean(torch.sum(torch.abs(news_rep - newsId_rep), dim=-1)))
        g_loss += sim_loss
        return g_loss

    def cal_user_d_loss(self, news_rep, user_rep, userId_rep):
        real_out = self.discriminator_user(news_rep, user_rep)
        fake_out = self.discriminator_user(news_rep, userId_rep)
        logit = real_out - fake_out
        # d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(),  0 , 1),
        #                                                     torch.flatten(torch.ones_like(real_out), 0, 1))
        d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(real_out))
        return d_loss

    def cal_user_g_loss(self, news_rep, user_rep, userId_rep):
        g_out = self.discriminator_user(news_rep, userId_rep)
        d_out = self.discriminator_user(news_rep, user_rep)
        logit = g_out - d_out
        # g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(), 0 , 1),
        #                                                     torch.flatten(torch.ones_like(g_out), 0, 1))
        g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(g_out))
        sim_loss = 0.1 * torch.mean(torch.square(user_rep - userId_rep))
        # sim_loss = 0.1 * (torch.mean(torch.sum(torch.abs(user_rep - userId_rep), dim=-1)))
        g_loss += sim_loss
        return g_loss

    def forward(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

        # userId/ newsId
        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot学习(Id)
        # loss_zeroshot_news, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # loss_zeroshot_user, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot学习(Att)
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        # loss_zeroshot_user, user_att_rep, La, Lc, Ld = self.zeroshot_user_tower(user_rep, user_att_rep, user_type_index.to(self.device))
        # 暂时不优化冷新闻
        # loss_zeroshot_news = torch.tensor(0)

        user_rep_gan = torch.flatten(user_rep.repeat(1, news_rep.shape[1], 1), 0, 1)
        user_att_rep_gan = torch.flatten(user_att_rep.unsqueeze(1).repeat(1, news_rep.shape[1], 1), 0, 1)
        news_rep_gan = torch.flatten(news_rep, 0, 1)
        # newsId_rep_gan = torch.flatten(newsId_rep, 0, 1)

        # # NEWS_gan_loss
        # news_d_loss = self.cal_news_d_loss(user_rep_gan, news_rep_gan, newsId_rep_gan)
        # news_g_loss = self.cal_news_g_loss(user_rep_gan, news_rep_gan, newsId_rep_gan)
        # news_gan_loss = news_d_loss + news_g_loss

        # User_gan_loss
        user_d_loss = self.cal_user_d_loss(news_rep_gan, user_rep_gan, user_att_rep_gan)
        user_g_loss = self.cal_user_g_loss(news_rep_gan, user_rep_gan, user_att_rep_gan)
        user_gan_loss = user_d_loss + user_g_loss


        # loss_zeroshot_news, loss_zeroshot_user = 0, 0
        user_rep = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        
        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        

        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, userId_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(torch.flatten(news_type_index.to(self.device), 0, 1).unsqueeze(1) == 0,
        #                    torch.flatten(newsId_rep, 0, 1),
        #                    torch.flatten(news_rep, 0, 1)).view(self.args.batch_size, self.args.sample_size, -1)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        # score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)

        return score, user_gan_loss

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # _, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # _, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        user_rep = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)

        # 预测得分
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        score = torch.sigmoid(score)

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, userId_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        # score = torch.sigmoid(score)
        return score
