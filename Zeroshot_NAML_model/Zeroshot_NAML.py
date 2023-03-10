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
        # ????????????
        category_embedding = self.embedding_layer1(category_index.to(torch.int64))
        category_rep = self.fc1(category_embedding)
        category_rep = F.dropout(category_rep, p=self.dropout_prob, training=self.training)
        #category_rep = self.norm1(category_rep)
        # ???????????????
        subcategory_embedding = self.embedding_layer2(subcategory_index.to(torch.int64))
        subcategory_rep = self.fc2(subcategory_embedding)
        subcategory_rep = F.dropout(subcategory_rep, p=self.dropout_prob, training=self.training)
        #subcategory_rep = self.norm2(subcategory_rep)
        # ????????????
        word_rep = self.cnn(word_embedding)
        word_rep = F.dropout(word_rep, p=self.dropout_prob, training=self.training)
        #word_rep = self.norm3(word_rep)
        # ???????????????
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


class Zeroshot_NAML(torch.nn.Module):
    def __init__(self, args, entity_embedding, relation_embedding, news_entity_dict, entity_adj, relation_adj,
                 news_title_word_index, word_embedding, news_category_index, news_subcategory_index, device):
        super(Zeroshot_NAML, self).__init__()
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

    def get_user_news_rep(self, candidate_news_index, user_clicked_news_index):
        # ????????????
        candidate_news_word_embedding = self.word_embedding[self.news_title_word_dict[candidate_news_index]].to(self.device)
        user_clicked_news_word_embedding = self.word_embedding[self.news_title_word_dict[user_clicked_news_index]].to(self.device)
        # ????????????
        candidate_news_category_index = torch.IntTensor(self.news_category_dict[np.array(candidate_news_index)]).to(self.device)
        user_clicked_news_category_index = torch.IntTensor(self.news_category_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)
        # ???????????????
        candidate_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(candidate_news_index.cpu())]).to(self.device)
        user_clicked_news_subcategory_index = torch.IntTensor(self.news_subcategory_dict[np.array(user_clicked_news_index.cpu())]).to(self.device)

        # ???????????????
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

        # ???????????????
        user_rep = None
        for i in range(self.args.batch_size):
            # ????????????????????????
            clicked_news_word_embedding_one = user_clicked_news_word_embedding[i, :, :, :]
            clicked_news_word_embedding_one = clicked_news_word_embedding_one.squeeze()
            # ??????????????????index
            clicked_news_category_index = user_clicked_news_category_index[i, :]
            # ?????????????????????index
            clicked_news_subcategory_index = user_clicked_news_subcategory_index[i, :]
            # ????????????
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


    def forward(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index):
        # ??????????????????
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

        # userId/ newsId
        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot??????(Id)
        # loss_zeroshot_news, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # loss_zeroshot_user, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot??????(Att)
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        loss_zeroshot_user, _, La, Lc, Ld = self.zeroshot_user_tower(user_rep, user_att_rep, user_type_index.to(self.device))
        # ????????????????????????
        loss_zeroshot_news = torch.tensor(0)

        #################### ???????????? ####################
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsId_rep * userId_rep.unsqueeze(1), dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(),
        #                    user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        # score = torch.sum(news_rep * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)
        score = torch.sum(news_rep * user_rep.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)
        return score, loss_zeroshot_news, loss_zeroshot_user, La, Lc, Ld

    def test(self, candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index ):
        # ??????????????????
        user_rep, news_rep, news_feature_list = self.get_user_news_rep(candidate_newsindex, user_clicked_newsindex)

        # newsId_rep = self.newsId_encoder(candidate_newsindex.to(self.device))
        # userId_rep = self.userId_encoder(user_index.to(self.device))

        # zeroshot??????(Id)
        # _, newsId_rep = self.zeroshot_news_tower(news_rep, newsId_rep, news_feature_list, news_type_index.to(self.device))
        # _, userId_rep = self.zeroshot_user_tower(user_rep, userId_rep, user_type_index.to(self.device))

        # zeroshot??????(Att)
        user_att_rep = self.get_user_att_rep(user_clicked_newsindex)
        _, user_rep_update, _, _, _ = self.zeroshot_user_tower(user_rep, user_att_rep, user_type_index.to(self.device))

        # ????????????
        # DNN
        # score = self.predict_layer(news_rep, user_rep.repeat(1, news_rep.shape[1], 1),
        #                            newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))

        # DOT
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score_id = torch.sum(newsId_rep * userId_rep.unsqueeze(1), dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # DNN + DOT
        # score_id = self.predict_layer(newsId_rep, userId_rep.unsqueeze(1).repeat(1, newsId_rep.shape[1], 1))
        # score_semantic = torch.sum(news_rep * user_rep, dim=-1).view(self.args.batch_size, -1)
        # score = score_id + score_semantic

        # replace
        # user = torch.where(user_type_index.to(self.device).unsqueeze(1) == 0, user_att_rep.squeeze(), user_rep.squeeze()).view(self.args.batch_size, 1, -1)
        # news = torch.where(news_type_index.to(self.device).unsqueeze(-1) == 0, newsId_rep, news_rep)
        # score = self.predict_layer(user.repeat(1, news.shape[1], 1), news)
        score = torch.sum(news_rep * user_rep_update.view(self.args.batch_size, 1, -1), dim=-1).view(self.args.batch_size, -1)
        score = torch.sigmoid(score)
        return score