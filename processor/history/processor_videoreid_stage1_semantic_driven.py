import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import einops


def do_train_stage1(cfg,
                    model,
                    train_loader_stage1,
                    optimizer,
                    scheduler,
                    local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    camids = []
    with torch.no_grad():
        for n_iter, (vids, pids, camid) in enumerate(train_loader_stage1):
            print(n_iter)
            vids = vids.to(device)
            target = pids.to(device)
            camid = camid.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(vids, target, get_image=True)
                for i, img_feat, c in zip(target, image_feature, camid):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
                    camids.append(c)
        labels_list = torch.stack(labels, dim=0).cuda()  # N
        camids_list = torch.stack(camids, dim=0).cuda().to(torch.int32)
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i * batch:(i + 1) * batch]
            else:
                b_list = iter_list[i * batch:num_image]

            target = labels_list[b_list]
            camid = camids_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                text_features_with_special = model(label = target, get_text_with_special = True, cam_label = camid)
                text_feature_only_shared = model(label=target, get_text_only_shared=True)
            text_features_with_special_patches = einops.rearrange(text_features_with_special, '(n m) d -> n m d', m=4)
            image_features_patches = einops.rearrange(image_features, '(n m) d -> n m d', m=4)
            camid_patches = einops.rearrange(camid, '(n m)-> n m', m=4)
            loss_patch = None
            for text_features_with_special_patch, image_features_patch, camid_patch in \
                zip(text_features_with_special_patches, image_features_patches, camid_patches):
                loss_i2t_patch = xent(text_features_with_special_patch, image_features_patch, camid_patch, camid_patch)
                loss_t2i_patch = xent(image_features_patch, text_features_with_special_patch, camid_patch, camid_patch)
                if loss_patch is None:
                    loss_patch = loss_i2t_patch + loss_t2i_patch
                else:
                    loss_patch += (loss_i2t_patch + loss_t2i_patch)

            loss_i2t = xent(image_features, text_feature_only_shared, target, target)
            loss_t2i = xent(text_feature_only_shared, image_features, target, target)

            # loss_i2t

            # target_with_camid
            # print(cfg.MODEL.T2I_I2T_WEIGHT_PATCH)
            loss = loss_i2t + loss_t2i + loss_patch * cfg.MODEL.T2I_I2T_WEIGHT_PATCH / 16
            # loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), vids.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
