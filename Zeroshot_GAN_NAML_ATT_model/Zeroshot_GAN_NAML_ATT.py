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

class newsatt_encoder(torch.nn.Module):
    def __init__(self, args):
        super(newsatt_encoder, self).__init__()
        self.args = args
        self.embedding_layer1 = nn.Embedding(self.args.category_num, embedding_dim=self.args.category_embedding_dim)
        self.embedding_layer2 = nn.Embedding(self.args.subcategory_num, embedding_dim=self.args.subcategory_embedding_dim)
        self.news_embedding = nn.Embedding(self.args.news_num, self.args.embedding_dim)

        self.fc1 = nn.Linear(self.args.category_embedding_dim, self.args.attention_dim * self.args.attention_heads, bias=True)
        self.fc2 = nn.Linear(self.args.subcategory_embedding_dim, self.args.attention_dim * self.args.attention_heads, bias=True)
        self.fc3 = nn.Linear(self.args.embedding_dim, self.args.attention_dim * self.args.attention_heads)

        self.attention = Additive_Attention(self.args.query_vector_dim, self.args.attention_dim * self.args.attention_heads)
        self.dropout_prob = 0.2

    def forward(self, candidate_newsindex, category_index, subcategory_index):
        # 主题表征
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)

        # 副主题表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)

        # id表征
        news_embedding = self.news_embedding(candidate_newsindex)
        newsId_rep = self.fc3(news_embedding)
        newsId_rep = F.dropout(newsId_rep, p=self.dropout_prob, training=self.training)

        # 附加注意力
        newsatt_rep = torch.cat([newsId_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        newsatt_rep = torch.tanh(self.attention(newsatt_rep))
        newsatt_rep = F.dropout(newsatt_rep, p=self.dropout_prob, training=self.training)
        return newsatt_rep

class useratt_encoder(torch.nn.Module):
    def __init__(self, args):
        super(useratt_encoder, self).__init__()
        self.args = args
        self.news_encoder = newsatt_encoder(args)
        self.user_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)
        self.fc = nn.Linear(self.args.embedding_dim, self.args.attention_dim * self.args.attention_heads)

        self.user_attention = Additive_Attention(self.args.query_vector_dim, self.args.attention_dim * self.args.attention_heads)
        self.dropout_prob = 0.2

    def forward(self, user_index, news_index, category_index, subcategory_index):
        news_rep = self.news_encoder(news_index, category_index, subcategory_index).unsqueeze(0)
         # id表征
        user_embedding = self.user_embedding(user_index).unsqueeze(0).unsqueeze(0)
        userId_rep = self.fc(user_embedding)
        userId_rep = F.dropout(userId_rep, p=self.dropout_prob, training=self.training)

        user_rep = torch.tanh(self.user_attention(torch.cat([userId_rep, news_rep], -2)))
        useratt_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return useratt_rep

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
        # category_rep = self.norm1(category_rep)
        # 副主题表征
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        # subcategory_rep = self.norm2(subcategory_rep)
        # 单词表征
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        # word_rep = self.norm3(word_rep)
        # 附加注意力
        news_rep = torch.cat([word_rep.unsqueeze(1), category_rep.unsqueeze(1), subcategory_rep.unsqueeze(1)], dim=1)
        news_rep = self.news_attention(news_rep)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
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
        user_rep = self.user_attention(news_rep.unsqueeze(0))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        # user_rep = self.norm2(user_rep)
        return user_rep


class Zeroshot_GAN_NAML_ATT(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(Zeroshot_GAN_NAML_ATT, self).__init__()
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



        self.newsatt_encoder = newsatt_encoder(args)

        self.useratt_encoder = useratt_encoder(args)
        self.predict_layer = predict_id_layer(args)

        # zeroshot学习
        self.zeroshot_news_tower = zeroshot_news_simple_tower(args)
        self.zeroshot_user_tower = zeroshot_user_simple_tower(args)

        # build discriminator and generator
        self.discriminator_news = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)
        self.discriminator_user = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)

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
    

    def get_user_news_att_rep(self, candidate_news_index, user_index, user_clicked_news_index):
                # 新闻主题
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # 新闻副主题
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # 新闻编码器
        news_rep = None
        for i in range(self.args.sample_size):
            news_index = candidate_news_index[:, i].to(self.device)
            news_category_index = candidate_news_category_index[:, i]
            news_subcategory_index = candidate_news_subcategory_index[:, i]
            news_rep_one = self.newsatt_encoder(news_index ,news_category_index, news_subcategory_index)
            if i == 0:
                news_rep = news_rep_one.unsqueeze(1)
            else:
                news_rep = torch.cat([news_rep, news_rep_one.unsqueeze(1)], dim=1)
                
        # 用户编码器
        user_rep = None
        for i in range(self.args.batch_size):
            user_index_one = user_index[i].to(self.device)
            clicked_news_index = user_clicked_news_index[i, :10].to(self.device)
            # 点击新闻主题index
            clicked_news_category_index = user_clicked_news_category_index[i, :10]
            # 点击新闻副主题index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :10]
            # 用户表征
            user_rep_one = self.useratt_encoder(user_index_one, clicked_news_index, clicked_news_category_index, clicked_news_subcategory_index).unsqueeze(0)
            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return news_rep, user_rep

    def cal_news_d_loss(self, user_rep, news_rep, newsatt_rep):
        real_out = self.discriminator_news(user_rep, news_rep)
        fake_out = self.discriminator_news(user_rep, newsatt_rep)
        logit = real_out - fake_out
        # d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(),  0 , 1),
        #                                                     torch.flatten(torch.ones_like(real_out), 0, 1))
        d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(real_out))
        return d_loss

    def cal_news_g_loss(self, user_rep, news_rep, newsatt_rep):
        g_out = self.discriminator_news(user_rep, newsatt_rep)
        d_out = self.discriminator_news(user_rep, news_rep)
        logit = g_out - d_out
        # g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(), 0 , 1),
        #                                                     torch.flatten(torch.ones_like(g_out), 0, 1))
        g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(g_out))
        sim_loss = 0.1 * torch.mean(torch.square(news_rep - newsatt_rep))
        # sim_loss = 0.1 * (torch.mean(torch.sum(torch.abs(news_rep - newsId_rep), dim=-1)))
        g_loss += sim_loss
        return g_loss

    def cal_user_d_loss(self, news_rep, user_rep, useratt_rep):
        real_out = self.discriminator_user(news_rep, user_rep)
        fake_out = self.discriminator_user(news_rep, useratt_rep)
        logit = real_out - fake_out
        # d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(),  0 , 1),
        #                                                     torch.flatten(torch.ones_like(real_out), 0, 1))
        d_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(real_out))
        return d_loss

    def cal_user_g_loss(self, news_rep, user_rep, useratt_rep):
        g_out = self.discriminator_user(news_rep, useratt_rep)
        d_out = self.discriminator_user(news_rep, user_rep)
        logit = g_out - d_out
        # g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(torch.flatten(logit.float(), 0 , 1),
        #                                                     torch.flatten(torch.ones_like(g_out), 0, 1))
        g_loss = (1.0 - 0.1) * torch.nn.BCEWithLogitsLoss()(logit.float(),
                                                            torch.ones_like(g_out))
        sim_loss = 0.1 * torch.mean(torch.square(user_rep - useratt_rep))
        # sim_loss = 0.1 * (torch.mean(torch.sum(torch.abs(user_rep - userId_rep), dim=-1)))
        g_loss += sim_loss
        return g_loss

    def forward(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)
         # 新闻用户属性表征
        newsatt_rep, useratt_rep = self.get_user_news_att_rep(candidate_newsindex, user_index, user_clicked_newsindex)

        user_rep_gan = torch.flatten(user_rep.repeat(1, news_rep.shape[1], 1), 0, 1)
        news_rep_gan = torch.flatten(news_rep, 0, 1)
        useratt_rep_gan = torch.flatten(useratt_rep.repeat(1, news_rep.shape[1], 1), 0, 1)
        newsatt_rep_gan = torch.flatten(newsatt_rep, 0, 1)

        # NEWS_gan_loss
        news_d_loss = self.cal_news_d_loss(user_rep_gan, news_rep_gan, newsatt_rep_gan)
        news_g_loss = self.cal_news_g_loss(user_rep_gan, news_rep_gan, newsatt_rep_gan)
        news_gan_loss = news_d_loss + news_g_loss

        # User_gan_loss
        user_d_loss = self.cal_user_d_loss(news_rep_gan, user_rep_gan, useratt_rep_gan)
        user_g_loss = self.cal_user_g_loss(news_rep_gan, user_rep_gan, useratt_rep_gan)
        user_gan_loss = user_d_loss + user_g_loss

        # zeroshot学习
        loss_zeroshot_news, newsatt_rep = self.zeroshot_news_tower(news_rep, newsatt_rep, news_feature_list, news_type_index.to(self.device))
        loss_zeroshot_user, useratt_rep = self.zeroshot_user_tower(user_rep, useratt_rep.squeeze(), user_type_index.to(self.device))

        user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, useratt_rep.squeeze(),
                           user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsatt_rep, news_rep)
        score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)

        # loss_zeroshot_news, loss_zeroshot_user = 0, 0

        # 预测得分
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsatt_rep, useratt_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsatt_rep * useratt_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsatt_rep, useratt_rep.repeat(1, newsatt_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        return score, news_gan_loss, user_gan_loss, loss_zeroshot_news, loss_zeroshot_user

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # 新闻用户表征
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)
        # 新闻用户属性表征
        newsatt_rep, useratt_rep = self.get_user_news_att_rep(candidate_newsindex, user_index, user_clicked_newsindex)

        # zeroshot学习
        _, newsatt_rep = self.zeroshot_news_tower(news_rep, newsatt_rep, news_feature_list, news_type_index.to(self.device))
        _, useratt_rep = self.zeroshot_user_tower(user_rep, useratt_rep.squeeze(), user_type_index.to(self.device))

        user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, useratt_rep.squeeze(),
                           user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsatt_rep, news_rep)
        score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)

        # 预测得分
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsatt_rep * useratt_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsatt_rep, useratt_rep.repeat(1, newsatt_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        score = torch.sigmoid(score)
        return score
