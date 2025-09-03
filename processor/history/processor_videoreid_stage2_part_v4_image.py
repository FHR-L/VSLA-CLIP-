import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
import einops
from tqdm import tqdm

from utils.test_video_reid import test, _eval_format_logger


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query,
             local_rank,
             optimizer_stage1):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    xent_query = SupConLoss(device, temperature=cfg.MODEL.TEMPERATURE)

    # torch.save(model.state_dict(),
    #            os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'initial_weight.pth'))
    # assert 1 < 0
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    print("collect all text features")
    with torch.no_grad():
        # model.without_prompt_(True)
        # model.without_dat_(True)
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
                # print(text_feature.shape)
                # assert 1 < 0
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
    # model.without_prompt_(False)
    # model.without_dat_(False)

    # model.change_stage(stage1=False)
    print(text_features[:5])
    print("text features has been collected, begin to train!")
    best_rank_1 = 0.0
    best_mAP =0.0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            # print(model.prompt_learner.text_shared_prompts[0])
            # TODO
            target_view = target_view
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            # print(target_view)
            # print(vid)
            # assert 1 < 0
            camera_id = target_cam.to(device)
            if cfg.MODEL.PBP_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.PBP_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                img = img.unsqueeze(2)
                score, feat, image_features, part_features_query, part_scores, part_features_split = model(
                    x=img,
                    label=target,
                    cam_label=target_cam,
                    view_label=target_view)
                text_feature = model(label=target, get_text=True)
                loss_i2t = xent(image_features, text_feature, target, target)
                loss_t2i = xent(text_feature, image_features, target, target)
                logits = image_features @ text_features.t()
                # [img_feature_last, img_feature, img_feature_proj]
                loss_query_split = None
                if part_features_split is not None and cfg.MODEL.PART_ALIGN_LOSS_WEIGHT != 0.:
                    # part_features_split [batch_size, num_query, dim]
                    batch_size, num_query, _ = part_features_split.shape
                    part_features_query_list = [part_features_query[i] for i in range(part_features_query.shape[0])]
                    part_features_split_list = [part_features_split[i] for i in range(part_features_split.shape[0])]
                    part_target = torch.tensor([i+1 for i in range(num_query)]).to(device)
                    for part_features_query_, part_features_split_, c_id in zip(part_features_query_list, part_features_split_list, camera_id):
                        # print(part_features_query.shape, part_features_split.shape)
                        if int(c_id) == 0:
                            if loss_query_split is None:
                                loss_query_split = xent_query(part_features_query_, part_features_split_, part_target, part_target)
                            else:
                                loss_query_split += xent_query(part_features_query_, part_features_split_, part_target, part_target)
                    loss_query_split = loss_query_split / len(part_features_query_list)

                cross_platform_align_loss = None
                if part_features_query is not None and cfg.MODEL.CROSS_PLATFORM_ALIGN_LOSS_WEIGHT != 0.:
                    loss_count = 0.
                    batch_size, num_query, _ = part_features_split.shape
                    part_target = torch.tensor([i + 1 for i in range(num_query)]).to(device)
                    p_cam_id = einops.rearrange(camera_id, '(p k) -> p k', k=4)
                    p_part_features_query = einops.rearrange(part_features_query, '(p k) nq d -> p k nq d', k=4)
                    p = p_part_features_query.shape[0]
                    for i in range(p):
                        cam_id = p_cam_id[i]
                        part_feat = p_part_features_query[i]
                        for m in range(4):
                            for n in range(4):
                                if cam_id[m] == cam_id[n]:
                                    continue
                                else:
                                    feat1 = part_feat[m]
                                    feat2 = part_feat[n]
                                    if cross_platform_align_loss is None:
                                        # print(feat1.shape)
                                        # print(feat2.shape)
                                        # print(part_target)
                                        # assert 1 < 0
                                        cross_platform_align_loss = xent_query(feat1, feat2, part_target, part_target)
                                        loss_count += 1.
                                    else:
                                        cross_platform_align_loss += xent_query(feat1, feat2, part_target, part_target)
                                        loss_count += 1.
                    cross_platform_align_loss = cross_platform_align_loss / loss_count



                # print(target)
                loss = loss_fn(score, feat, target, camera_id, logits, loss_t2i, loss_i2t,
                               part_features_query=part_features_query,
                               part_scores=part_scores,
                               loss_query_split=loss_query_split,
                               cross_platform_align_loss=cross_platform_align_loss)

                # 更新样本中心 #
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                assert 1 < 0
            else:
                use_gpu = True
                cmc, mAP, ranks = do_inference_image(cfg, model, val_loader, num_query)
                ptr = "mAP: {:.2%}".format(mAP)
                for r in [1, 5, 10]:
                    ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                logger.info(ptr)
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_mAP_best.pth'))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_image(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    pd = tqdm(total=len(val_loader), ncols=120, leave=False)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        pd.update(1)
        with torch.no_grad():
            img = img.to(device)
            camids = None
            target_view = None
            img = img.unsqueeze(2)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
    pd.close()

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc, mAP, cmc
