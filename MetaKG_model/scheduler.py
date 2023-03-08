from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from utils.measure import *
import torch.nn.functional as F
import random

class Scheduler(nn.Module):
    def __init__(self, N, grad_indexes, use_deepsets=True):
        super(Scheduler, self).__init__()
        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.grad_lstm_2 = nn.LSTM(N, 10, 1, bidirectional=True)
        self.grad_indexes = grad_indexes
        self.use_deepsets = use_deepsets
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 40
        if use_deepsets:
            self.h = nn.Sequential(nn.Linear(input_dim, 20), nn.Tanh(), nn.Linear(20, 10))
            self.fc1 = nn.Linear(input_dim + 10, 20)
        else:
            self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, loss, input):
        grad_output, (_, _) = self.grad_lstm(input[0].reshape(1, len(input[0]), -1))
        grad_output = grad_output.sum(0)
        print('=====')
        loss_output, (_, _) = self.loss_lstm(loss.unsqueeze(0))
        loss_output = loss_output.sum(0)
        x = torch.cat((grad_output, loss_output), dim=1)
        print(x.shape)
        if self.use_deepsets:
            x_C = (torch.sum(x, dim=1).unsqueeze(1) - x) / (len(x) - 1)
            print(x_C.shape)
            x_C_mapping = self.h(x_C)
            x = torch.cat((x, x_C_mapping), dim=1)
            z = torch.tanh(self.fc1(x))
        else:
            z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        p = prob.detach().cpu().numpy()
        if len(np.where(p > 0)[0]) < size:
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)),
                                       p=p/np.sum(p),
                                       size=size,
                                       replace=replace)
            actions = [torch.tensor(x).to(self.device) for x in actions]
        return torch.LongTensor(actions)

    def compute_loss(self, candidate_newsindex, user_index, user_clicked_newsindex, labels, model):
        # 先不随机生成，前25为support，后25为query
        support_size = int(candidate_newsindex.shape[0] / 2)
        support_user_index, support_candidate_newsindex, support_labels = user_index[:support_size], \
                                                                          candidate_newsindex[:support_size], \
                                                                          labels[:support_size]

        query_user_index, query_candidate_newsindex, query_labels = user_index[support_size:], \
                                                                    candidate_newsindex[support_size:], \
                                                                    labels[support_size:]

        loss_meta_query, scores, _, _ = model.forward_meta(support_user_index, support_candidate_newsindex, support_labels,
                                                           query_user_index, query_candidate_newsindex, query_labels, fast_weights=None)
        task_losses = loss_meta_query

        try:
            rec_auc = roc_auc_score(query_labels.cpu().numpy(), F.softmax(scores.cpu(), dim=1).detach().numpy())
        except ValueError:
            rec_auc = 0.5
        task_auc = rec_auc
        return task_losses, task_auc

    def get_weight(self, candidate_newsindex, user_index, user_clicked_newsindex, labels, model, pt):
        input_embedding_norm = []
        # 先不随机生成，前25为support，后25为query
        support_size = int(candidate_newsindex.shape[0]/2)
        support_user_index, support_candidate_newsindex, support_labels = user_index[:support_size], \
                                                                          candidate_newsindex[:support_size], \
                                                                          labels[:support_size]

        query_user_index, query_candidate_newsindex, query_labels = user_index[support_size:], \
                                                                    candidate_newsindex[support_size:], \
                                                                    labels[support_size:]

        loss_support, _, _, _ = model.forward(support_user_index, support_candidate_newsindex, support_labels)
        loss_query, _, _, _ = model.forward(query_user_index, query_candidate_newsindex, query_labels)

        task_losses = torch.cat([loss_support, loss_query], dim = 0)

        fast_weights = OrderedDict(model.named_parameters())
        fast_weights.pop('relation_embedding.weight')
        fast_weights.pop('entity_embedding.weight')

        task_grad_support = []
        task_grad_query = []

        for i in range(task_losses.shape[0]):
            print(i)
            task_grad = []
            task_grad_norm = []
            if i < support_size:
                task_grad_support.append(torch.autograd.grad(loss_support[i], fast_weights.values(),
                                                             retain_graph=True, create_graph=False))
                for j in range(len(task_grad_support[-1])):
                    task_grad.append(task_grad_support[i][j])
            else:
                task_grad_query.append(torch.autograd.grad(loss_query[i-support_size], fast_weights.values(),
                                                           retain_graph=True, create_graph=False))
                for j in range(len(task_grad_support[-1])):
                    task_grad.append(task_grad_query[i-support_size][j])

            for j in range(len(task_grad)):
                task_grad_norm.append(task_grad[j].norm())

            task_grad_norm = torch.stack(task_grad_norm)
            input_embedding_norm.append(task_grad_norm.detach())

        # torch.cuda.empty_cache()
        task_layer_inputs = [torch.stack(input_embedding_norm).to(self.device)]
        weight = self.forward(task_losses.unsqueeze(1), task_layer_inputs).to(self.device)
        # weight = weight.detach()
        return task_losses, weight
