from bio_embeddings.embed import ProtTransT5XLU50Embedder
from models import Tran_PPI
import torch
from util import get_dataloader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from timm.utils import accuracy
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, matthews_corrcoef, confusion_matrix, average_precision_score
import util.utils as utils
import torch.distributed as dist
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(
        'trans-ppi training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--width', default=320, type=int)
    parser.add_argument('--relative_pos', action="store_true")
    parser.add_argument('--key_analysis', action="store_true")
    parser.add_argument('--train_idpd', action="store_true")
    parser.add_argument('--method', default="self_attention", type=str)
    parser.add_argument('--embed_method', default="pt5", type=str)
    
    #optimizer
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    
    parser.add_argument('--folder_num',default=1,type=int)
    parser.add_argument('--output_dir', default='', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--add_fea', default=None, type=str,
                        help='path for the add features, empty for no add')
    parser.add_argument('--ckpt', default=None, type=str,
                        help='checkpoint path')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=8e-4, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=5e-4, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=5e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=15, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=0, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument("--local_rank", default=0, type=int)
    
    parser.add_argument('--seed', default=88, type=int)
    
    return parser   

def specificity(y_true, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    return tn / (tn + fp)

@torch.no_grad()            
def evaluate(model: torch.nn.Module,embedder,criterion,data_loader,add_fea):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    add_fea = True
    preds_item_list = []
    labels_item_list = []
    for idx, full_data in enumerate(metric_logger.log_every(data_loader, 1, header)):
        if add_fea:
            human_seq, virus_seq, human_pro_add, virus_pro_add, label = full_data
            label = label.cuda()
            human_emb = list(embedder.embed_many(human_seq))
            virus_emb = list(embedder.embed_many(virus_seq))
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb,human_pro_add,virus_pro_add)
                loss = criterion(preds,label)
        else:
            human_seq, virus_seq, label = full_data
            label = label.cuda()
            human_emb = list(embedder.embed_many(human_seq))
            virus_emb = list(embedder.embed_many(virus_seq))
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb)
                loss = criterion(preds,label)
        preds = torch.sigmoid(preds)
        preds[torch.where(preds>=0.5)] = 1.
        preds[torch.where(preds<0.5)] = 0.
        preds_item_list.append(preds)
        labels_item_list.append(label)
        acc1 = accuracy_score(preds.cpu().numpy().copy(), label.cpu().numpy().copy())
        batch_size = label.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        
    preds_item_list = torch.cat(preds_item_list,dim=0)
    labels_item_list = torch.cat(labels_item_list,dim=0)
    all_item_list = torch.cat([preds_item_list,labels_item_list],dim=1)
    all_list = [None for _ in range(torch.distributed.get_world_size())]
    dist.barrier()
    dist.all_gather_object(all_list, all_item_list)
    preds_list = [item[:,0].cpu().numpy().copy() for item in all_list]
    labels_list = [item[:,1].cpu().numpy().copy() for item in all_list]
    preds_list = np.concatenate(preds_list,axis=0)
    labels_list = np.concatenate(labels_list,axis=0)
    acc_all = accuracy_score(labels_list,preds_list)
    auc_all = roc_auc_score(labels_list,preds_list)
    auprc_all = average_precision_score(labels_list, preds_list)
    pre_score_all = precision_score(labels_list,preds_list)
    re_score_all = recall_score(labels_list,preds_list)
    f1 = f1_score(labels_list,preds_list)
    mcc = matthews_corrcoef(labels_list,preds_list)
    sp = specificity(labels_list,preds_list)
    print("auc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} mcc:{:.4f} auprc:{:.4f} sp:{:.4f} acc: {top1.global_avg:.4f} loss: {losses.global_avg:.4f} \n"
                    .format(auc_all, pre_score_all, re_score_all, f1, mcc, auprc_all, sp, top1=metric_logger.acc1, losses=metric_logger.loss))
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
    #     .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    #     preds = torch.sigmoid(preds)
    #     print(preds)
    #     print(label)
    #     auc_pred = np.append(auc_pred,preds.cpu().numpy().copy())
    #     auc_label = np.append(auc_label,label.cpu().numpy().copy())
    #     loss_avg += loss.item()
    #     preds[torch.where(preds>=0.5)] = 1.
    #     preds[torch.where(preds<0.5)] = 0.
    #     print(preds)
    #     correct_num += (preds==label).sum()
    #     total_num += len(human_seq)
    # loss_avg /= total_iter
    # acc_total = correct_num / total_num
    # auc_score = roc_auc_score(auc_label, auc_pred)
    # print("acc: {} auc: {} loss: {}".format(acc_total, auc_score, loss_avg))
    # return acc_total

def main(args):
    
    utils.init_distributed_mode(args)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    if args.embed_method == "pt5":
        embedder = ProtTransT5XLU50Embedder(half_precision_model=True)
    else:
        raise ValueError('no such embedding method')
    model = Tran_PPI(embedder.embedding_dimension, args.width, args.relative_pos, args.key_analysis, args.method)
    _, test_loader = get_dataloader(args.batch_size,args.folder_num,args.num_workers,args.width,args.add_fea,args.train_idpd)
    checkpoint = torch.load(args.ckpt)
    msg = model.load_state_dict(checkpoint['model'],strict=False)
    print(msg)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    evaluate(model,embedder,loss_fn,test_loader,args.add_fea)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'trans-ppi evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)