import os

import torch
from nets.deeplabv3_training import (CE_Loss, mseloss, Focal_Loss, Dice_loss,
                                     weights_init, relax_loss, depth_loss, relax_loss)
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, \
    fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss       = 0
    total_mse_loss   = 0
    total_depth_loss = 0
    total_focal_loss = 0
    total_dice_loss  = 0

    val_loss        = 0
    val_mse_score     = 0
    val_depth_loss      = 0
    val_focal_loss   = 0
    val_dice_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, dans, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                dans    = dans.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()

        if epoch < 70:
            from torch.cuda.amp import autocast
            with autocast():
                pred_dan, x = model_train(imgs)
                loss = CE_Loss(pred_dan, dans, weights, num_classes)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        from torch.cuda.amp import autocast
        with autocast():
            #   前向传播
            pred_dan, x = model_train(imgs)
            # x = model_train(imgs)
            #   计算ROI损失
            loss = relax_loss(x, pngs)
            # loss = depth_loss(x, pngs)
            # mse = mseloss(x, pngs)
            # loss = loss + mse
            with torch.no_grad():
                # _f = CE_Loss(pred_dan, dans, weights, num_classes)
                _mse_loss = mseloss(x, pngs)
                # _depth = depth_loss(x, pngs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss      += loss.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                # 'mse_score'   : total_mse_loss / (iteration + 1),
                                # 'depth_loss': total_depth_loss / (iteration + 1),
                                # 'focal_loss': total_focal_loss / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, dans, labels = batch
        with torch.no_grad():
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                dans = dans.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)
            #   前向传播
            pred_dan, x     = model_train(imgs)
            # x     = model_train(imgs)
            #   计算损失
            loss = relax_loss(x, pngs)
            # loss = depth_loss(x, pngs)
            #
            val_depth_loss += loss.item()
            #
            # mse = mseloss(x, pngs)

            #   计算胆囊损失
            # focal = CE_Loss(pred_dan, dans, weights, num_classes)
            # focal = 0

            # loss = loss + mse

            val_loss    += loss.item()
            # val_mse_score += mse.item()
            # val_focal_loss += focal.item()


            if local_rank == 0:
                pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                    # 'mse_score'   : val_mse_score / (iteration + 1),
                                    # 'depth_loss': val_depth_loss / (iteration + 1),
                                    # 'focal_loss': val_focal_loss / (iteration + 1),
                                    'lr'        : get_lr(optimizer)})
                pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        # eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' % (epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))