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
    def __init__(self, word_dim, num_filters, window_sizes, query_vector_dim):
        super(news_encoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=num_filters,
                              kernel_size=(window_sizes, word_dim),
                              padding=(int((window_sizes - 1) / 2), 0))
        self.news_attention = QueryAttention(query_vector_dim, num_filters)
        self.dropout_prob = 0.3

    def forward(self, word_embedding, user_embedding):
        # ????????????
        word_embedding = self.conv(word_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        word_embedding = F.dropout(word_embedding, p=self.dropout_prob, training=self.training)
        word_embedding = torch.relu(word_embedding.transpose(2,1))
        # ???????????????
        if len(user_embedding.shape) == 2:
            user_embedding = user_embedding.unsqueeze(1)
        news_rep = self.news_attention(user_embedding, word_embedding)
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        return news_rep

class user_encoder(torch.nn.Module):
    def __init__(self, word_dim, num_filters, window_sizes, query_vector_dim):
        super(user_encoder, self).__init__()
        self.news_encoder = news_encoder(word_dim, num_filters, window_sizes, query_vector_dim)
        #self.user_attention = Additive_Attention(query_vector_dim, num_filters)
        self.user_attention = QueryAttention(query_vector_dim, num_filters)
        self.dropout_prob = 0.3

    def forward(self, word_embedding, user_embedding, user_embeding_two):
        news_rep = self.news_encoder(word_embedding, user_embedding.unsqueeze(0).repeat(50, 1, 1))
        news_rep = F.dropout(news_rep, p=self.dropout_prob, training=self.training)
        user_rep = self.user_attention(user_embeding_two.unsqueeze(0).unsqueeze(0),  news_rep.unsqueeze(0))
        user_rep = F.dropout(user_rep, p=self.dropout_prob, training=self.training)
        return user_rep


class user_embedding_encoder(torch.nn.Module):
    def __init__(self, args):
        super(user_embedding_encoder, self).__init__()
        self.args = args
        self.user_embedding = nn.Embedding(self.args.user_num, self.args.embedding_dim)
        self.fc1 = nn.Linear(self.args.embedding_dim, self.args.query_vector_dim, bias=True)
        self.fc2 = nn.Linear(self.args.embedding_dim, self.args.query_vector_dim, bias=True)

    def forward(self, user_index):
        user_embedding = self.user_embedding(user_index)
        user_vector = torch.relu(self.fc1(user_embedding))
        user_vector_2 = torch.relu(self.fc2(user_embedding))
        return user_vector, user_vector_2


class Zeroshot_GAN_NPA(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(Zeroshot_GAN_NPA, self).__init__()
        self.args = args
        self.device = device

        # no_embedding
        self.word_embedding = word_embedding
        self.entity_embedding = entity_embedding
        self.relation_embedding = relation_embedding

        # ??????ID???????????????
        self.user_embedding_encoder = user_embedding_encoder(args)

        # NPA
        self.news_encoder = news_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)
        self.user_encoder = user_encoder(self.args.word_embedding_dim, self.args.cnn_num_filters,
                                         self.args.cnn_window_sizes, self.args.query_vector_dim)

        # build discriminator and generator
        self.discriminator_news = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)
        self.discriminator_user = discriminator(self.args.attention_heads * self.args.attention_dim,
                                                self.args.attention_heads * self.args.attention_dim,
                                                self.args.embedding_dim)


        # ???????????????ID??????
        # self.newsId_encoder = newsId_encoder(args)
        # self.userId_encoder = userId_encoder(args)
        
        # ???????????????????????????
        self.userAtt_encoder = user_att_encoder(args)

        # zeroshot??????
        self.zeroshot_news_tower = zeroshot_news_simple_tower(args)
        self.zeroshot_user_tower = zeroshot_user_simple_tower(args)

        # predict 
        # self.predict_layer = predict_id_layer(args)
        # dict
        self.news_title_word_dict = news_title_word_index
        self.news_category_dict = news_category_index
        self.news_subcategory_dict = news_subcategory_index
        self.entity_adj = entity_adj
        self.relation_adj = relation_adj
        self.news_entity_dict = news_entity_dict


    def get_user_news_rep(self, user_index,  candidate_news_index, user_clicked_news_index):
        # ????????????
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        
        # user query
        user_vector, user_vector_2 = self.user_embedding_encoder(user_index.to(self.device))

        ## ???????????????
        news_rep = None
        for i in range(self.args.sample_size):
            title_word_embedding_one = candidate_news_word_embedding[:, i, :]
            news_rep_one = self.news_encoder(title_word_embedding_one, user_vector)
            news_rep_one = news_rep_one.unsqueeze(1)
            if i == 0:
                news_rep = news_rep_one
            else:
                news_rep = torch.cat([news_rep, news_rep_one], dim=1)

        # ???????????????
        user_rep = None
        for i in range(self.args.batch_size):
            clicked_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_word_embedding_one = clicked_word_embedding_one.squeeze()
            # ????????????
            user_vector_one = user_vector[i, :]
            user_vector_one_2 = user_vector_2[i, :]
            user_rep_one = self.user_encoder(clicked_word_embedding_one, user_vector_one, user_vector_one_2).unsqueeze(0)

            if i == 0:
                user_rep = user_rep_one
            else:
                user_rep = torch.cat([user_rep, user_rep_one], dim=0)
        return user_rep, news_rep, None


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
        # ??????????????????
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(user_index,  candidate_newsindex, user_clicked_newsindex)

        # userId/ newsId
        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot??????(Id)
        # loss_zeroshot_news, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # loss_zeroshot_user, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot??????(Att)
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        # loss_zeroshot_user, user_att_rep, La, Lc, Ld = self.zeroshot_user_tower(user_rep, user_att_rep, user_type_index.to(self.device))
        # ????????????????????????
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
        
        # ????????????
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        

        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, userId_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(torch.flatten(news_type_index.to(self.device), 0, 1).unsqueeze(1) == 0,
        #                    torch.flatten(newsId_rep, 0, 1),
        #                    torch.flatten(news_rep, 0, 1)).view(self.args.batch_size, self.args.sample_size, -1)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        # score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)

        return score, user_gan_loss

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # ??????????????????
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(user_index,  candidate_newsindex, user_clicked_newsindex)

        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # _, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # _, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        user_rep = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)

        # ????????????
        score = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        score = torch.sigmoid(score)

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, userId_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = torch.sum(news * user, dim=-1).view(self.args.batch_size, -1)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        # score = torch.sigmoid(score)
        return score
