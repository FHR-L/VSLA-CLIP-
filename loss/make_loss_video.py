import torch.nn.functional as F
from torch import nn

from loss.center_loss import CenterLoss
from loss.losses_video import CrossEntropyLabelSmooth, TripletLoss, CameraAwareLoss, TripletLossCameraAware
from loss.supcontrast import SupConLoss


def make_loss(cfg, num_classes):
    feat_dim = 2048
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target, target_cam, i2tscore=None, loss_t2i=None, loss_i2t=None, concat_feat=None,
                  part_features_query=None,
                  part_scores=None,
                  loss_query_split=None,
                  cross_platform_align_loss=None):

        criterions = {
            'xent': nn.CrossEntropyLoss(),
            'htri': TripletLoss(margin=cfg.MODEL.ID_LOSS_MARGIN, distance=cfg.TEST.DISTANCE),
            'cal': CameraAwareLoss(margin=cfg.MODEL.CAMERA_AWARE_LOSS_MARGIN),
        }
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
            if cfg.MODEL.IF_LABELSMOOTH == 'on':
                if isinstance(score, list):
                    ID_LOSS = [xent(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = xent(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = [criterions['htri'](feats, target, target_cam) for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS =criterions['htri'](feat, target, target_cam)

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if i2tscore != None and cfg.MODEL.I2T_LOSS_WEIGHT != 0:
                    I2TLOSS = xent(i2tscore, target)
                    loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                if cfg.MODEL.CAMERA_AWARE_LOSS_WEIGHT != 0 and concat_feat is not None:
                    if isinstance(concat_feat, list):
                        CAMERA_AWARE_LOSS = [criterions['cal'](f, target, target_cam) for f in concat_feat[0:]]
                        CAMERA_AWARE_LOSS = sum(CAMERA_AWARE_LOSS)
                    else:
                        CAMERA_AWARE_LOSS = criterions['cal'](concat_feat, target, target_cam)
                    loss = loss + cfg.MODEL.CAMERA_AWARE_LOSS_WEIGHT * CAMERA_AWARE_LOSS

                if part_features_query is not None:
                    part_features_query_list = [part_features_query[:, i] for i in range(part_features_query.shape[1])]
                    TRI_LOSS_Part = [criterions['htri'](feats, target, target_cam) for feats in part_features_query_list[0:]]
                    TRI_LOSS_Part = sum(TRI_LOSS_Part)
                    # print(TRI_LOSS_Part)
                    loss = loss + cfg.MODEL.TRI_PART_LOSS_WEIGHT * TRI_LOSS_Part

                if part_scores is not None:
                    if isinstance(part_scores, list):
                        ID_LOSS_PARTS = [xent(scor, target) for scor in part_scores[0:]]
                        ID_LOSS_PARTS = sum(ID_LOSS_PARTS)
                    else:
                        ID_LOSS_PARTS = xent(part_scores, target)
                    # print(ID_LOSS_PARTS)
                    loss = loss + cfg.MODEL.ID_LOSS_PARTS_WEIGHT * ID_LOSS_PARTS

                if loss_query_split is not None and cfg.MODEL.PART_ALIGN_LOSS_WEIGHT != 0.:
                    # print('loss_query_split:', loss_query_split)
                    loss = loss + cfg.MODEL.PART_ALIGN_LOSS_WEIGHT * loss_query_split

                if cross_platform_align_loss is not None and cfg.MODEL.CROSS_PLATFORM_ALIGN_LOSS_WEIGHT != 0.:
                    # print(cross_platform_align_loss)
                    loss = loss + cfg.MODEL.CROSS_PLATFORM_ALIGN_LOSS_WEIGHT * cross_platform_align_loss

                loss = loss + cfg.MODEL.I2T_WEIGHT * loss_i2t + cfg.MODEL.T2I_WEIGHT * loss_t2i

                return loss
            else:
                if isinstance(score, list):
                    ID_LOSS = [F.cross_entropy(scor, target) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS)
                else:
                    ID_LOSS = F.cross_entropy(score, target)

                if isinstance(feat, list):
                    TRI_LOSS = [criterions['htri'](feats, target)[0] for feats in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)
                else:
                    TRI_LOSS = criterions['htri'](feat, target)[0]

                loss = cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS + cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS

                if i2tscore != None and cfg.MODEL.I2T_LOSS_WEIGHT != 0:
                    I2TLOSS = F.cross_entropy(i2tscore, target)
                    loss = cfg.MODEL.I2T_LOSS_WEIGHT * I2TLOSS + loss

                if cfg.MODEL.CAMERA_AWARE_LOSS_WEIGHT != 0:
                    loss = loss + cfg.MODEL.CAMERA_AWARE_LOSS_WEIGHT * criterions['cal'](feat, target, target_cam)

                loss = loss + cfg.MODEL.I2T_WEIGHT * loss_i2t + cfg.MODEL.T2I_WEIGHT * loss_t2i
                return loss
    return loss_func, center_criterion