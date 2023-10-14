from util import get_dataloader, train_one_epoch, evaluate
from models import Tran_PPI
from pathlib import Path
from timm.scheduler import create_scheduler
import torch
from bio_embeddings.embed import ProtTransT5XLU50Embedder
from timm.utils import NativeScaler
import util.utils as utils
import numpy as np

import os

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
    parser.add_argument('--resume', default=None, type=str,
                        help='resume path')
    
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
    parser.add_argument('--min-lr', type=float, default=1e-4, metavar='LR',
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

def main(args):
    print(args)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    utils.init_distributed_mode(args)
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    
    start_epoch = args.start_epoch
    train_loader, test_loader = get_dataloader(args.batch_size,args.folder_num,args.num_workers,args.width,args.add_fea,args.train_idpd)
    
    print("Creating model")
    
    if args.embed_method == "pt5":
        embedder = ProtTransT5XLU50Embedder(half_precision_model=True)
    else:
        raise ValueError('no such embedding method')
    model = Tran_PPI(embedder.embedding_dimension, args.width, args.relative_pos, args.key_analysis, args.method)
    print(model)
    if args.resume:
        checkpoint = torch.load(args.resume,map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'],strict=False)
        print(msg)
        start_epoch = checkpoint['epoch'] + 1
    model = model.to(device)
    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    model_without_ddp = model.module
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        loss_scaler.load_state_dict(checkpoint['scaler'])
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = utils.Focal_Loss(alpha=0.9,reduction="mean")
    
    torch.distributed.barrier()
    
    
    print("start training")
    max_acc = 0.0
    if args.output_dir:
        os.makedirs(args.output_dir,exist_ok=True)
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)
        lr_scheduler.step(epoch)
        print("epoch:{}".format(epoch+1))
        train_one_epoch(model,embedder,loss_fn,train_loader,optimizer,loss_scaler,args.add_fea,args)
        test_stats = evaluate(model,embedder,loss_fn,test_loader,args.add_fea,epoch,args)
        if max_acc < test_stats["acc1"]:
            max_acc = test_stats["acc1"]
            if args.output_dir:
                checkpoint_path = os.path.join(args.output_dir,"checkpoint{}.pth".format(epoch))
                utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'trans-ppi training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
