from MRNN_model.MRNN_Train_Test import *
from NRMS_model.NRMS_Train_Test import *
from NAML_model.NAML_Train_Test import *
from MNN4Rec_model.MNN4Rec_Train_Test import *
from LSTUR_model.LSTUR_Train_Test import *
from DKN_model.DKN_Train_Test import *
from GAN_exp1_model.GAN_exp1_Train_Test import *
from LightGCN_model.LightGCN_Train_Test import *
from LightGCN_VAE_model.LightGCN_VAE_Train_Test import *
from FM_model.FM_Train_Test import *
from MNN4Rec_update_model.MNN4Rec_update_Train_Test import *
from MetaKG_model.MetaKG_Train_Test import *
from NPA_model.NPA_Train_Test import *
from Zeroshot_MRNN_model.Zeroshot_MRNN_Train_Test import *
from Zeroshot_NRMS_model.Zeroshot_NRMS_Train_Test import *
from Zeroshot_IDvae_model.Zeroshot_IDvae_Train_Test import *
from Zeroshot_base_model.Zeroshot_base_Train_Test import *
from Zeroshot_baseID_model.Zeroshot_baseID_Train_Test import *
from Zeroshot_GAN_MRNN_model.Zeroshot_GAN_MRNN_Train_Test import *
from Zeroshot_NAML_model.Zeroshot_NAML_Train_Test import *
from Zeroshot_GAN_MRNN_ATT_model.Zeroshot_GAN_MRNN_ATT_Train_Test import *
from Zeroshot_GAN_NRMS_model.Zeroshot_GAN_NRMS_Train_Test import *
from Zeroshot_GAN_NRMS_ATT_model.Zeroshot_GAN_NRMS_ATT_Train_Test import *
from Zeroshot_GAN_NAML_model.Zeroshot_GAN_NAML_Train_Test import *
from Zeroshot_GAN_NAML_ATT_model.Zeroshot_GAN_NAML_ATT_Train_Test import *
from DataLoad_MIND import load_data
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_data_mode', type=int, default=0)
    parser.add_argument('--news_data_mode', type=int, default=1)
    parser.add_argument('--mode', type=str, default='MNN4Rec')
    parser.add_argument('--epoch', type=int, default= 60)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5)
    parser.add_argument('--optimizer_type', type=str, default='adam')
    parser.add_argument('--loss_type', type=str, default='cross_entropy')
    parser.add_argument('--checkpoint_dir', type=str, default='./out/save_model/', help='??????????????????')

    parser.add_argument('--user_num', type = int, default=711222)
    parser.add_argument('--user_clicked_num', type=int, default=50)
    parser.add_argument('--warm_user_num', type=int, default=539069, help='????????????')
    parser.add_argument('--cold_user_num', type=int, default=172153, help='????????????')
    parser.add_argument('--news_num', type=int, default=101528, help='????????????')
    parser.add_argument('--warm_news_num', type=int, default=79547, help='????????????')
    parser.add_argument('--cold_news_num', type=int, default=21981, help='????????????')
    parser.add_argument('--category_num', type=int, default=19, help='??????????????????')
    parser.add_argument('--subcategory_num', type=int, default=286, help='?????????????????????')
    parser.add_argument('--word_num', type=int, default=65829, help='????????????')
    parser.add_argument('--news_entity_num', type=int, default=43432, help='????????????????????????')
    parser.add_argument('--total_entity_num', type=int, default=141627, help='?????????????????????')
    parser.add_argument('--total_relation_num', type=int, default=458, help='?????????????????????')
    parser.add_argument('--news_entity_size', type=int, default=20, help='??????????????????????????????')
    parser.add_argument('--title_word_size', type=int, default=65, help='????????????????????????????????????')
    parser.add_argument('--entity_neigh_num', type=int, default=5, help='??????????????????')

    # MRNN
    parser.add_argument('--attention_heads', type=int, default=20, help='????????????????????????')
    parser.add_argument('--num_units', type=int, default=20, help='???????????????????????????')
    parser.add_argument('--attention_dim', type=int, default=20, help='?????????????????????')
    parser.add_argument('--embedding_dim', type=int, default=100, help='?????????????????????')
    parser.add_argument('--title_embedding_dim', type=int, default=400, help='????????????????????????')
    parser.add_argument('--word_embedding_dim', type=int, default=300, help='??????????????????')
    parser.add_argument('--entity_embedding_dim', type=int, default=100, help='??????????????????')
    parser.add_argument('--category_embedding_dim', type=int, default=100, help='????????????')
    parser.add_argument('--subcategory_embedding_dim', type=int, default=100, help='???????????????')
    parser.add_argument('--query_vector_dim', type=int, default=200, help='??????????????????')

    parser.add_argument('--ripplenet_n_hop', type=int, default=2, help='maximum hops')
    parser.add_argument('--ripplenet_n_memory', type=int, default=5, help='size of ripple set for each hop')
    # DKN??????
    parser.add_argument('--kcnn_num_filters', type=int, default=50, help='???????????????')
    parser.add_argument('--kcnn_window_sizes', type=list, default=[2, 3, 4], help='????????????')
    parser.add_argument('--use_context', type=bool, default=None, help='???????????????')
    # NAML??????
    parser.add_argument('--cnn_num_filters', type=int, default=400, help='???????????????')
    parser.add_argument('--cnn_window_sizes', type=int, default=3, help='????????????')
    parser.add_argument('--drop_prob', type=bool, default=0.2, help='????????????')
    # LSTUR??????
    parser.add_argument('--long_short_term_method', type=str, default='ini', help='ini or con')
    parser.add_argument('--lstur_num_filters', type=int, default=300, help='???????????????')
    parser.add_argument('--lstur_window_sizes', type=int, default=3, help='????????????')
    parser.add_argument('--masking_probability', type=int, default=0.3, help='????????????')
    # LightGCN
    parser.add_argument("--lgn_layers", type=int, default=3, help="the number of layers in GCN")
    parser.add_argument("--keep_prob", type=float, default=0.6, help="the batch size for bpr loss training procedure")
    parser.add_argument("--dropout", type=float, default=0, help="using the dropout or not")
    parser.add_argument('--l2', type=float, default=1e-5, help='l2 regularization weight')
    # MNN4Rec??????
    parser.add_argument('--topk_implicit', type=int, default=5, help='?????????????????????')
    parser.add_argument('--topk_explicit', type=int, default=5, help='?????????????????????')
    parser.add_argument('--use_news_relation', type=bool, default=True, help='????????????????????????')
    # MetaKG
    parser.add_argument('--meta_update_lr', type=float, default=0.001, help='meta update learning rate')
    parser.add_argument('--scheduler_lr', type=float, default=0.001, help='scheduler learning rate')
    parser.add_argument('--num_inner_update', type=int, default=2, help='number of inner update')
    parser.add_argument('--meta_batch_size', type=int, default=50, help='meta batch size')
    parser.add_argument('--n_hops', type=int, default=10, help='gcn hop')
    parser.add_argument("--mess_dropout", type=bool, default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of node dropout")
    parser.add_argument('--select_idx', type=bool, default=False, help='??????????????????')
    return parser.parse_args()

def main(path, device):
    args = parse_args()
    data = load_data(args,path)
    if args.mode == "MRNN":
        model = MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NRMS":
        model = NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NAML":
        model = NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MNN4Rec":
        model = MNN4Rec_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LSTUR":
        model = LSTUR_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "DKN":
        model = DKN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "GAN_exp1":
        model = GAN_exp1_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LightGCN":
        model = LightGCN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "LightGCN_VAE":
        model = LightGCN_VAE_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "FM":
        model = FM_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MNN4Rec_update":
        model = MNN4Rec_update_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "MetaKG":
        model = MetaKG_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "NPA":
        model = NPA_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_MRNN":
        model = Zeroshot_MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NRMS":
        model = Zeroshot_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_IDvae":
        model = Zeroshot_IDvae_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_base":
        model = Zeroshot_base_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_baseID":
        model = Zeroshot_baseID_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_MRNN":
        model = Zeroshot_GAN_MRNN_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_NAML":
        model = Zeroshot_NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_MRNN_ATT":
        model = Zeroshot_GAN_MRNN_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NAML":
        model = Zeroshot_GAN_NAML_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NAML_ATT":
        model = Zeroshot_GAN_NAML_ATT_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NRMS":
        model = Zeroshot_GAN_NRMS_Train_Test(args, data, device)
        model.Test_load()
    if args.mode == "Zeroshot_GAN_NRMS_ATT":
        model = Zeroshot_GAN_NRMS_ATT_Train_Test(args, data, device)
        model.Test_load()
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    path = os.path.dirname(os.getcwd())
    main(path, device)
