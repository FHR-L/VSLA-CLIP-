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

from utils.test_video_reid import test, _eval_format_logger


def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             query_loader,
             gallery_loader,
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
    # for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage2):
    #     print(n_iter)
    # print(len(train_loader_stage2))

    model.eval()
    id_list = []
    cam_list = []
    feat_list = []
    for n_iter, (img, vid, target_cam) in enumerate(train_loader_stage2):
        # print(model.prompt_learner.text_shared_prompts[0])
        # TODO
        print(n_iter)
        if n_iter > 7:
            break
        target_view = target_cam
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
            feat = model(
                x=img,
                label=target,
                cam_label=target_cam,
                view_label=target_view)
            id_list.append(vid)
            cam_list.append(target_cam)
            feat_list.append(feat)

    torch.save(id_list, cfg.OUTPUT_DIR + '/id_list.pth')
    torch.save(cam_list, cfg.OUTPUT_DIR + '/cam_list.pth')
    torch.save(feat_list, cfg.OUTPUT_DIR + '/feat_list.pth')