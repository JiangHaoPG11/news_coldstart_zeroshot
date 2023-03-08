from MetaKG_model.MetaKG import MetaKG
from MetaKG_model.MetaKG_Trainer import Trainer
from MetaKG_model.scheduler import Scheduler


import torch

class MetaKG_Train_Test():
    def __init__(self, args, data, device):
        train_dataloader, test_dataloader, vaild_dataloader, \
        news_title_embedding, entity_adj, relation_adj, entity_dict, kg_env, news_entity_dict, entity_news_dict, user_click_dict, \
        news_title_word_index, news_category_index, news_subcategory_index, category_news_dict, subcategory_news_dict, word_embedding, \
        neibor_embedding, neibor_num, entity_embedding, relation_embedding, ripple_set, \
        vailddata_size, traindata_size, testdata_size, label_test, bound_test = data

        self.args = args

        self.MetaKG_model = MetaKG(args, entity_embedding, relation_embedding,
                                   news_entity_dict, entity_adj, relation_adj,
                                   news_title_word_index, word_embedding, news_category_index,
                                   news_subcategory_index, user_click_dict, device).to(device)
        optimizer_MetaKG = torch.optim.Adam(self.MetaKG_model.parameters(), lr=0.0001)
        names_weights_copy, indexes = self.get_net_parameter_dict(self.MetaKG_model.named_parameters(), device)
        self.scheduler = Scheduler(len(names_weights_copy), grad_indexes=indexes).to(device)
        scheduler_optimizer = torch.optim.Adam(self.scheduler.parameters(), lr=args.scheduler_lr)
        self.trainer = Trainer(args, self.MetaKG_model, optimizer_MetaKG, self.scheduler, scheduler_optimizer, data)

    def get_net_parameter_dict(self, params, device):
        param_dict = dict()
        indexes = []
        for i, (name, param) in enumerate(params):
            if param.requires_grad:
                param_dict[name] = param.to(device)
                indexes.append(i)
        return param_dict, indexes

    def Train(self):
        print('training begining ...')
        # AnchorKG_model.train()
        self.trainer.train()

    def Test(self):
        self.trainer.test()

    def Test_load(self):
        self.MetaKG_model.load_state_dict(torch.load(self.args.checkpoint_dir + 'checkpoint-' +
                                                     self.args.mode +
                                                     '-epochfinal.pth'))
        self.trainer.test()
