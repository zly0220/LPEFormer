from bio_embeddings.embed import ProtTransT5XLU50Embedder
from models import Tran_PPI
import torch
from torch.utils.data import DataLoader
from util import Seq_Pair_dataset
import argparse
import pandas as pd
import numpy as np
from copy import deepcopy
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser(
        'trans-ppi training and evaluation script', add_help=False)

    parser.add_argument('--width', default=320, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--csv_file', default="data_full/train_all.csv", type=str)
    parser.add_argument('--ckpt', default="outputs/test11_train_idpd/checkpoint17.pth", type=str)
    parser.add_argument('--add_fea', default=None, type=str,
                        help='path for the add features, empty for no add')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int)
    
    
    return parser 

def main(args):
    dataset = Seq_Pair_dataset(args.csv_file, args.width, args.add_fea, return_pro_name=True)
    dataloader = DataLoader(dataset,args.batch_size,shuffle=False,num_workers=8,pin_memory=True)
    embedder = ProtTransT5XLU50Embedder(half_precision_model=True)
    model = Tran_PPI(width=args.width, relative_pos=True)
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    print("Model params {} are not loaded".format(msg.missing_keys))
    print("State-dict params {} are not used".format(msg.unexpected_keys))
    model.eval().cuda()
    pd_list = []
    with torch.no_grad():
        for idx, full_data in tqdm(enumerate(dataloader)):
            human_seq, virus_seq, human_pro_add, virus_pro_add, human_pro_name, virus_pro_name, label = full_data
            batch_size = len(human_pro_name)
            human_pro_name = np.asarray(human_pro_name)
            virus_pro_name = np.asarray(virus_pro_name)
            label = label.reshape((batch_size)).numpy()
            human_emb = list(embedder.embed_many(human_seq))
            virus_emb = list(embedder.embed_many(virus_seq))
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb,human_pro_add,virus_pro_add)
                preds = torch.sigmoid(preds)
                preds_save = deepcopy(preds).reshape((batch_size)).cpu().numpy()
                preds[torch.where(preds>=0.5)] = 1.
                preds[torch.where(preds<0.5)] = 0.
                preds = preds.reshape((batch_size)).cpu().numpy()
            property_pred = np.zeros((batch_size,))
            index = np.where((preds==1.) & (preds!=label))
            property_pred[index] = 1.
            index = np.where((label==1.) & (preds!=label))
            property_pred[index] = 2.
            index = np.where((preds==1.) | (label==1.))
            data_save = {'human_pro':human_pro_name[index],
                         'virus_pro':virus_pro_name[index],
                         'preds':preds_save[index],
                         'property':property_pred[index]}
            pd_data = pd.DataFrame(data_save)
            pd_list.append(pd_data)
            # if idx == 10:
            #     break
        pd_all = pd.concat(pd_list,ignore_index=True).to_csv("test.csv")
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'trans-ppi prediction script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)