import os
import time

from config import cfg
import argparse
from datasets.make_video_dataloader import make_dataloader
from model.make_model_vsla_part_ import make_model
from utils.test_video_reid import test
from utils.logger import setup_logger
import warnings
warnings.filterwarnings("ignore")

# CUDA_VISIBLE_DEVICES=0 python

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    (train_loader_stage2, train_loader_stage1,
     query_loader, gallery_loader,
     query_loader_a2g, gallery_loader_a2g,
     num_classes, num_query, camera_num) = make_dataloader(cfg)

    # TODO
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=0)
    model.load_param(cfg.TEST.WEIGHT)

    computation_complexity = True
    if computation_complexity:
        device = "cuda"
        model.to(device)
        model.eval()
        from thop import profile
        import torch

        x = torch.randn(1, 3, 8, 256, 128).to(device)
        label = torch.randn(16).long()
        cam_label = torch.randn(1).long().clamp(0, 1)
        view_label = torch.randn(1).long().clamp(1, 1)
        # print(x.shape)
        # assert 0
        # baseline macs 68.778
        # Part macs 68.929388544
        # VSL++ 68.929486848
        #

        macs, params = profile(model, inputs=(x, None, False))
        print(macs / (1000 * 1000 * 1000))
        print(params)

        total_params = 0
        trainable_params = 0

        # 遍历模型的所有参数
        model = model.image_encoder
        for name, param in model.named_parameters():
            num_params = param.numel()  # 当前参数的数量
            total_params += num_params

            # 判断是否为可训练参数
            if param.requires_grad:
                trainable_params += num_params
        print(f"总参数数量: {total_params}")
        print(f"可训练参数数量: {trainable_params}")
        # assert 0

        start_time = time.time()
        # label = torch.randn(1).long()
        # cam_label = torch.randn(1).long().clamp(0, 1).to(device)
        # view_label = torch.randn(1).long().clamp(1, 1).to(device)
        # x = torch.randn(1, 8, 3, 256, 128).to(device)
        # 49.53
        for i in range(1000):
            model(x, None, False)
        end_time = time.time()

        # 计算运行时间
        elapsed_time = end_time - start_time
        print(f"{1000 / elapsed_time:.6f} FPS")
        assert 0


    use_gpu = True
    cmc, mAP, ranks = test(model, query_loader, gallery_loader, use_gpu, cfg)
    ptr = "G2A mAP: {:.2%}".format(mAP)
    for r in ranks:
        ptr += " | R-{:<3}: {:.2%}".format(r, cmc[r - 1])
    logger.info(ptr)