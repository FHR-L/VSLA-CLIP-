import logging
import os
import time
import torch
import torch.nn as nn

from datasets.image.make_dataloader_clipreid import make_dataloader
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
            view_id = target_view.to(device)
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

                loss = loss_fn(score, feat, target, camera_id, logits, loss_t2i, loss_i2t,
                               part_features_query=part_features_query,
                               part_scores=part_scores,
                               loss_query_split=None,
                               cross_platform_align_loss=None)

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
                cmc, mAP, ranks = do_inference_image(cfg, model, val_loader, num_query, save_dist_mat=True)
                ptr = "mAP: {:.2%}".format(mAP)
                for r in [1, 5, 10]:
                    ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                logger.info(ptr)
                if mAP > best_mAP:
                    best_mAP = mAP
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_mAP_best.pth'))


                if cfg.DATASETS.NAMES == 'CARGO':
                    _, _, val_loader_aa, num_query_aa, _, _, _ = make_dataloader(
                        cfg, dataset_name='CARGO_AA', dataset_root='/home/Newdisk1/luowenlong/Datasets/CARGO')

                    cmc, mAP, ranks = do_inference_image(cfg, model, val_loader_aa, num_query_aa)
                    ptr = "AA mAP: {:.2%}".format(mAP)
                    for r in [1, 5, 10]:
                        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                    logger.info(ptr)

                    # AG
                    _, _, val_loader_ag, num_query_ag, _, _, _ = make_dataloader(
                        cfg, dataset_name='CARGO_AG', dataset_root='/home/Newdisk1/luowenlong/Datasets/CARGO')

                    cmc, mAP, ranks = do_inference_image(cfg, model, val_loader_ag, num_query_ag)
                    ptr = "AG mAP: {:.2%}".format(mAP)
                    for r in [1, 5, 10]:
                        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                    logger.info(ptr)

                    # GG
                    _, _, val_loader_gg, num_query_gg, _, _, _ = make_dataloader(
                        cfg, dataset_name='CARGO_GG', dataset_root='/home/Newdisk1/luowenlong/Datasets/CARGO')

                    cmc, mAP, ranks = do_inference_image(cfg, model, val_loader_gg, num_query_gg)
                    ptr = "GG mAP: {:.2%}".format(mAP)
                    for r in [1, 5, 10]:
                        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
                    logger.info(ptr)

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)


def do_inference_image(cfg,
                 model,
                 val_loader,
                 num_query,
                 save_dist_mat=False):
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
    view_list = []

    pd = tqdm(total=len(val_loader), ncols=120, leave=False)
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        pd.update(1)
        view_list.extend(target_view)
        with torch.no_grad():
            img = img.to(device)
            camids = None
            target_view = None
            img = img.unsqueeze(2)
            feat = model(img, cam_label=camids, view_label=target_view)
            # print(feat.shape)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    pd.close()

    cmc, mAP, dist_mat, _, _, q_pids, g_pids = evaluator.compute()
    if save_dist_mat:
        torch.save(dist_mat, cfg.OUTPUT_DIR + '/distmat.pth')
        torch.save(q_pids, cfg.OUTPUT_DIR + '/q_pids.pth')
        torch.save(g_pids, cfg.OUTPUT_DIR + '/g_pids.pth')
        torch.save(view_list, cfg.OUTPUT_DIR + '/view.pth')
        torch.save(evaluator.get_feat(), cfg.OUTPUT_DIR + '/feats.pth' )

    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc, mAP, cmc
