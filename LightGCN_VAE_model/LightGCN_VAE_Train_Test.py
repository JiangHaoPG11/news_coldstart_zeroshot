from LightGCN_VAE_model.LightGCN_VAE import LightGCN_VAE
from LightGCN_VAE_model.LightGCN_VAE_Trainer import Trainer
import torch

class LightGCN_VAE_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        LightGCN_VAE_model = LightGCN_VAE(args, news_title_embedding, entity_embedding, relation_embedding,
                 news_entity_dict, entity_adj, relation_adj, user_click_dict, news_title_word_index,
                 news_category_index, news_subcategory_index, device).to(device)
        optimizer_LightGCN_VAE = torch.optim.Adam(LightGCN_VAE_model.parameters(), lr=0.0001)
        self.trainer = Trainer(args, LightGCN_VAE_model, optimizer_LightGCN_VAE, data)
        self.LightGCN_VAE_model = LightGCN_VAE_model
        self.args = args

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def Test_load(self):
        self.LightGCN_VAE_model.load_state_dict(torch.load(self.args.checkpoint_dir + 'checkpoint-' + self.args.mode + '-epochfinal.pth'))
        self.trainer.test()