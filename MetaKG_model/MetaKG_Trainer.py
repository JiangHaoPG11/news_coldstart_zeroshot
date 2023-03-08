import os
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import no_grad
from utils.measure import *
from numpy import *
import torch
import torch.nn as nn

# torch.cuda.empty_cache()
class Trainer():
    def __init__(self, args, MetaKG, optimizer_MetaKG, scheduler, scheduler_optimizer, data):
        self.args = args
        self.MetaKG_model = MetaKG
        self.optimizer_MetaKG = optimizer_MetaKG
        self.scheduler = scheduler
        self.scheduler_optimizer = scheduler_optimizer

        self.save_period = 1
        self.vaild_period = 1
        self.train_dataloader = data[0]
        self.test_dataloader = data[1]
        self.vaild_dataloader = data[2]
        self.news_embedding = data[3]
        self.entity_dict = data[6]
        self.entity_embedding = data[12]
        self.vailddata_size = data[-5]
        self.traindata_size = data[-4]
        self.testdata_size = data[-3]
        self.label_test = data[-2]
        self.bound_test = data[-1]

        self.criterion = nn.CrossEntropyLoss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        #self.device = torch.device("cpu")

    def cal_auc(self, label, rec_score):
        try:
            auc = roc_auc_score(label.cpu().numpy(), F.softmax(rec_score.cpu(), dim=1).detach().numpy())
        except ValueError:
            auc = 0.5
        return auc

    def update_moving_avg(self, mavg, reward, count):
        return mavg + (reward.item() - mavg) / (count + 1)

    def optimize_model(self, rec_score, label):
        rec_loss = self.criterion(rec_score, torch.argmax(label, dim=1).to(self.device))
        self.optimizer_MetaKG.zero_grad()
        rec_loss.backward()
        #for name, param in self.model_recommender.named_parameters():
        #    print('%14s : %s' % (name, param.grad))
        self.optimizer_MetaKG.step()
        return rec_loss

    def _train_epoch(self, epoch):
        # self.MetaKG_model.train()
        loss_list = []
        auc_list = []
        moving_avg_reward = 0
        pbar = tqdm(total=self.traindata_size,  desc=f"Epoch {epoch}", ncols=100, leave=True, position=0)
        for data in self.train_dataloader:
            candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data
            selected_tasks_index = None
            loss_scheduler = 0
            index = 0
            pt = int(index / self.traindata_size) * 100
            index += 1
            if candidate_newsindex.shape[0] > self.args.meta_batch_size:
                if self.args.select_idx == True:
                    task_losses, weight_meta_batch = self.scheduler.get_weight(candidate_newsindex, user_index,
                                                                               user_clicked_newsindex, label,
                                                                               self.MetaKG_model, pt)
                    task_prob = torch.softmax(weight_meta_batch.reshape(-1), dim=-1)
                    selected_tasks_index = self.scheduler.sample_task(task_prob, self.args.meta_batch_size)
                else:
                    selected_tasks_index = np.random.choice(candidate_newsindex.shape[0],
                                                            self.args.meta_batch_size,
                                                            replace=False)
                candidate_newsindex_selected = candidate_newsindex[selected_tasks_index]
                user_index_selected = user_index[selected_tasks_index]
                user_clicked_newsindex_selected = user_clicked_newsindex[selected_tasks_index]
                labels_selected = label[selected_tasks_index]
                selected_losses, selected_auc = self.scheduler.compute_loss(candidate_newsindex_selected,
                                                                            user_index_selected,
                                                                            user_clicked_newsindex_selected,
                                                                            labels_selected, self.MetaKG_model)
            else:
                selected_losses, selected_auc = self.scheduler.compute_loss(candidate_newsindex,
                                                                            user_index,
                                                                            user_clicked_newsindex,
                                                                            label, self.MetaKG_model)
            meta_batch_loss = torch.mean(selected_losses)
            meta_batch_auc = mean(selected_auc)

            if self.args.select_idx == True:
                for index in selected_tasks_index:
                    loss_scheduler += self.scheduler.m.log_prob(index.to(self.device))

            reward = meta_batch_loss
            loss_scheduler = loss_scheduler * (reward - moving_avg_reward)
            moving_avg_reward = self.update_moving_avg(moving_avg_reward, reward, index)
            self.scheduler_optimizer.zero_grad()
            loss_scheduler.backward(retain_graph=True)
            self.scheduler_optimizer.step()

            self.optimizer_MetaKG.zero_grad()
            meta_batch_loss.backward()
            self.optimizer_MetaKG.step()

            loss_list.append(meta_batch_loss.cpu().item())
            auc_list.append(meta_batch_auc)

            pbar.update(self.args.batch_size)
            # print("----recommend loss：{}-----rec auc：{} ". format(str(torch.mean(rec_loss).cpu().item()), str(rec_auc)))

        pbar.close()
        return mean(loss_list), mean(auc_list)

    def _vaild_epoch(self):
        self.MetaKG_model.eval()
        rec_auc_list = []
        with no_grad():
            pbar = tqdm(total=self.vailddata_size)
            for data in self.vaild_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, label, user_type_index, news_type_index = data

                scores = self.MetaKG_model.test(user_index, candidate_newsindex)
                rec_auc = self.cal_auc(label, scores)
                rec_auc_list.append(rec_auc)
                pbar.update(self.args.batch_size)
            pbar.close()
        return mean(rec_auc_list)

    def _save_checkpoint(self, epoch):
        state_MetaKG_model = self.MetaKG_model.state_dict()
        filename_MetaKG = self.args.checkpoint_dir + ('checkpoint-MetaKG-epoch{}.pth'.format(epoch))
        torch.save(state_MetaKG_model, filename_MetaKG)

    def train(self):
        for epoch in range(1, self.args.epoch+1):
            rec_loss, rec_auc = self._train_epoch(epoch)
            print("epoch：{}----recommend loss：{}------rec auc：{}-------".
                  format(epoch, str(rec_loss), str(rec_auc)))

            if epoch % self.vaild_period == 10:
                print('start vaild ...')
                rec_auc = self._vaild_epoch()
                print("epoch：{}---vaild auc：{} ".format(epoch, str(rec_auc)))

            if epoch % self.save_period == 60:
                self._save_checkpoint(epoch)
        self._save_checkpoint('final')

    def test(self):
        print('start testing...')
        pbar = tqdm(total= self.testdata_size)
        self.MetaKG_model.eval()
        pred_label_list = []
        user_index_list = []
        user_type_list = []
        news_type_list = []
        candidate_newsindex_list = []
        with no_grad():
            for data in self.test_dataloader:
                candidate_newsindex, user_index, user_clicked_newsindex, user_type_index, news_type_index = data
                rec_score = self.MetaKG_model.test(user_index, candidate_newsindex)
                score = rec_score
                pred_label_list.extend(score.cpu().numpy())
                user_index_list.extend(user_index.cpu().numpy())

                user_type_list.extend(user_type_index.cpu().numpy())
                news_type_list.extend(news_type_index.cpu().numpy())
                candidate_newsindex_list.extend(candidate_newsindex.cpu().numpy()[:,0])
                pbar.update(self.args.batch_size)
            pred_label_list = np.vstack(pred_label_list)
            pbar.close()
        # 存储预测结果
        folder_path = '../predict/MetaKG/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        predict_df = pd.DataFrame()
        predict_df['user'] = user_index_list
        predict_df['user_type'] = user_type_list
        predict_df['candidate_news'] = candidate_newsindex_list
        predict_df['candidate_news_type'] = news_type_list
        predict_df['score'] = pred_label_list[:, 0]
        predict_df['label'] = self.label_test[:len(user_index_list)]
        predict_df.to_csv('MetaKG_predict.csv', index = False)

        test_AUC, test_MRR, test_nDCG5, test_nDCG10 = evaluate(pred_label_list, self.label_test, self.bound_test)
        print("test_AUC = %.4lf, test_MRR = %.4lf, test_nDCG5 = %.4lf, test_nDCG10 = %.4lf" %
              (test_AUC, test_MRR, test_nDCG5, test_nDCG10))
        print('================user====================')
        c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len, \
        w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len, = evaluate_warm_cold_u(pred_label_list, user_type_list, news_type_list,
                                                                       self.label_test, self.bound_test)
        print("c_AUC = %.4lf, c_MRR = %.4lf, c_nDCG5 = %.4lf, c_nDCG10 = %.4lf, c_len = %.4lf" %
              (c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len))
        print("w_AUC = %.4lf, w_MRR = %.4lf, w_nDCG5 = %.4lf, w_nDCG10 = %.4lf, w_len = %.4lf" %
              (w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len))
        print('================news====================')
        c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len, \
        w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len = evaluate_warm_cold_n(pred_label_list, user_type_list, news_type_list,
                                                                      self.label_test, self.bound_test)
        print("c_AUC = %.4lf, c_MRR = %.4lf, c_nDCG5 = %.4lf, c_nDCG10 = %.4lf, c_len = %.4lf" %
              (c_AUC, c_MRR, c_nDCG5, c_nDCG10, c_len))
        print("w_AUC = %.4lf, w_MRR = %.4lf, w_nDCG5 = %.4lf, w_nDCG10 = %.4lf, w_len = %.4lf" %
              (w_AUC, w_MRR, w_nDCG5, w_nDCG10, w_len))
        print('================news-user===============')
        cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10, cc_len, \
        cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10, cw_len, \
        wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10, wc_len, \
        ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10, ww_len = evaluate_warm_cold(pred_label_list, user_type_list, news_type_list, self.label_test, self.bound_test)

        print("cc_AUC = %.4lf, cc_MRR = %.4lf, cc_nDCG5 = %.4lf, cc_nDCG10 = %.4lf, cc_len = %.4lf" %
              (cc_AUC, cc_MRR, cc_nDCG5, cc_nDCG10, cc_len))
        print("cw_AUC = %.4lf, cw_MRR = %.4lf, cw_nDCG5 = %.4lf, cw_nDCG10 = %.4lf, cw_len = %.4lf" %
              (cw_AUC, cw_MRR, cw_nDCG5, cw_nDCG10, cw_len))
        print("wc_AUC = %.4lf, wc_MRR = %.4lf, wc_nDCG5 = %.4lf, wc_nDCG10 = %.4lf, wc_len = %.4lf" %
              (wc_AUC, wc_MRR, wc_nDCG5, wc_nDCG10, wc_len))
        print("ww_AUC = %.4lf, ww_MRR = %.4lf, ww_nDCG5 = %.4lf, w_nDCG10 = %.4lf, ww_len = %.4lf" %
              (ww_AUC, ww_MRR, ww_nDCG5, ww_nDCG10, ww_len))

