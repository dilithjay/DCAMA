r""" training (validation) code """
import torch.optim as optim
import torch.nn as nn
import torch

from model.DCAMA import DCAMA
from common.logger import Logger, AverageMeter
from common.evaluation import Evaluator
from common.config import parse_opts
from common import utils
from data.dataset import FSSDataset


def train(epoch, model, dataloader, optimizer, training):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    utils.fix_randseed(None) if training else utils.fix_randseed(0)
    model.train_mode() if training else model.eval()
    average_meter = AverageMeter(dataloader.dataset)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for idx, batch in enumerate(dataloader):

        # 1. forward pass
        if device == 'cuda':
            batch = utils.to_cuda(batch)
        logit_mask = model(batch['query_img'], batch['support_imgs'].squeeze(1), batch['support_masks'].squeeze(1))
        pred_mask = logit_mask.argmax(dim=1)

        # 2. Compute loss & update model parameters
        loss = model.compute_objective(logit_mask, batch['query_mask'])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 3. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss.detach().clone())
        average_meter.write_process(idx, len(dataloader), epoch, write_batch_idx=50)

    # Write evaluation results
    average_meter.write_result('Training' if training else 'Validation', epoch)
    avg_loss = utils.mean(average_meter.loss_buf)
    miou, fb_iou = average_meter.compute_iou()

    return avg_loss, miou, fb_iou


if __name__ == '__main__':

    # Arguments parsing
    args = parse_opts()

    # ddp backend initialization
    # torch.distributed.init_process_group(backend='nccl')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.set_device(args.local_rank)

    # Model initialization
    model = DCAMA().float()
    model.to(device)
    # model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
    #                                             find_unused_parameters=True)

    # Helper classes (for training) initialization
    optimizer = optim.SGD([{"params": model.parameters(), "lr": args.lr,
                            "momentum": 0.9, "weight_decay": args.lr/10, "nesterov": True}])
    Evaluator.initialize()
    Logger.initialize(args, training=True)
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    
    losses = []
    mious = []

    for i in range(5):
        # Dataset initialization
        FSSDataset.initialize(img_size=256, datapath=args.datapath, use_original_imgsize=False)
        dataloader_trn = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'trn')
        dataloader_val = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'val')

        # Train
        best_val_miou = float('-inf')
        best_val_loss = float('inf')
        for epoch in range(args.nepoch):
            trn_loss, trn_miou, trn_fb_iou = train(epoch, model, dataloader_trn, optimizer, training=True)

            with torch.no_grad():
                val_loss, val_miou, val_fb_iou = train(epoch, model, dataloader_val, optimizer, training=False)

            # Save the best model
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                Logger.save_model_miou(model, epoch, val_miou)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            Logger.tbd_writer.add_scalars('data/loss', {'trn_loss': trn_loss, 'val_loss': val_loss}, epoch)
            Logger.tbd_writer.add_scalars('data/miou', {'trn_miou': trn_miou, 'val_miou': val_miou}, epoch)
            Logger.tbd_writer.add_scalars('data/fb_iou', {'trn_fb_iou': trn_fb_iou, 'val_fb_iou': val_fb_iou}, epoch)
            Logger.tbd_writer.flush()
    
        Logger.tbd_writer.close()
        Logger.info(f'==================== Finished Training for Fold {i} ====================')

        losses.append(best_val_loss)
        mious.append(best_val_miou)
    
    print("Losses:", losses)
    print("mIoUs:", mious)