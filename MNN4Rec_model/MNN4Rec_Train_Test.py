from MNN4Rec_model.MNN4Rec import MNN4Rec
from MNN4Rec_model.MNN4Rec_Trainer import Trainer
import torch

class MNN4Rec_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        MNN4Rec_model = MNN4Rec(args, entity_embedding, relation_embedding,
                          news_entity_dict, entity_adj, relation_adj, news_title_word_index,
                          word_embedding, news_category_index, news_subcategory_index, device).to(device)
        optimizer_MNN4Rec = torch.optim.Adam(MNN4Rec_model.parameters(), lr=0.0001)
        self.trainer = Trainer(args, MNN4Rec_model, optimizer_MNN4Rec, data)
        self.MNN4Rec_model = MNN4Rec_model
        self.args = args

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def Test_load(self):
        self.MNN4Rec_model.load_state_dict(torch.load(self.args.checkpoint_dir + 'checkpoint-' + self.args.mode + '-epochfinal.pth'))
        self.trainer.test()