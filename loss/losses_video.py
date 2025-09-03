from __future__ import absolute_import

import einops
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.

    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = Variable(targets, requires_grad=False)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, target, _):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target:  ground truth labels with shape (batch_size)
        :return:
        """
        n = input.size(0)
        # Compute pairwise distance, replace by the official when merged
        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = - torch.matmul(input, input.t())
        else:
            raise NotImplementedError

        # For each anchor, find the hardest positive and negative
        # print(n)
        # print(dist)
        mask = target.expand(n, n).eq(target.expand(n, n).t()).float()
        # print(dist*mask - (1-mask))
        dist_ap, _ = torch.topk(dist*mask - (1-mask), dim=-1, k=1)
        # print(dist*(1-mask) + mask)
        dist_an, _ = torch.topk(dist*(1-mask) + mask, dim=-1, k=1, largest=False)
        # assert 1 <0


        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        # print(dist_an)
        # print(dist_ap)
        # print(y)
        # assert 1 < 0
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLossCameraAwareV2(nn.Module):
    """Triplet loss with hard positive/negative mining and camera-awareness.
    Args:
        margin (float): margin for triplet.
        distance (str): distance metric, options are ['cosine', 'euclidean'].
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLossCameraAwareV2, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, target, camera_id):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target: ground truth labels with shape (batch_size)
        :param camera_id: camera IDs with shape (batch_size)
        :return: triplet loss value
        """
        n = input.size(0)

        # Compute pairwise distance
        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = -torch.matmul(input, input.t())
        else:
            raise NotImplementedError("Currently only cosine distance is supported.")

        # Label match mask
        label_mask = target.expand(n, n).eq(target.expand(n, n).t()).float()

        # Camera ID match and mismatch masks
        same_camera_mask = camera_id.expand(n, n).eq(camera_id.expand(n, n).t()).float()
        diff_camera_mask = 1 - same_camera_mask

        # Valid positive pairs: same label and same camera ID
        valid_pos_mask = label_mask * same_camera_mask

        # Valid negative pairs: different labels and different camera IDs
        valid_neg_mask = (1 - label_mask) * diff_camera_mask

        # Hardest positive
        dist_ap, _ = torch.topk(dist * valid_pos_mask - (1 - valid_pos_mask) * 1e6, dim=-1, k=1)

        # Hardest negative
        dist_an, _ = torch.topk(dist * valid_neg_mask + (1 - valid_neg_mask) * 1e6, dim=-1, k=1, largest=False)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

def check_list_pattern(lst):
    # 检查列表长度是否是4的倍数
    if len(lst) % 4 != 0:
        return False

    # 遍历列表，每4个元素作为一组进行检查
    for i in range(0, len(lst), 4):
        if len(set(lst[i:i+4])) != 1:  # 检查每组是否只有一个唯一值
            return False

    return True

class CameraAwareLoss(nn.Module):
    """Triplet loss with hard positive/negative mining and camera-awareness.
    Args:
        margin (float): margin for triplet.
        distance (str): distance metric, options are ['cosine', 'euclidean'].
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(CameraAwareLoss, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        # print(f"camera aware loss margin:{margin}")

    def check_list_pattern(self, lst):
        lst = lst.tolist()
        if len(lst) % 4 != 0:
            return False

        for i in range(0, len(lst), 4):
            if len(set(lst[i:i + 4])) != 1:
                return False

        return True

    def forward(self, input, target, camera_id):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target: ground truth labels with shape (batch_size)
        :param camera_id: camera IDs with shape (batch_size)
        :return: triplet loss value
        """
        # print('camera aware')
        # print(target)
        assert self.check_list_pattern(target)
        feat_patch = einops.rearrange(input, '(b n) d -> b n d', n=4)
        target_patch = einops.rearrange(target, '(b n) -> b n', n=4)
        camera_id_patch = einops.rearrange(camera_id, '(b n) -> b n', n=4)
        feat_center = torch.mean(feat_patch, dim=1, keepdim=False)
        n = feat_patch.size(0)
        label_mask = target_patch[:,0].expand(n, n).eq(target_patch[:,0].expand(n, n).t()).float()
        valid_neg_mask = (1 - label_mask)

        if self.distance == 'cosine':
            feat_center = F.normalize(feat_center, dim=-1)
            dist = -torch.matmul(feat_center, feat_center.t())
        else:
            raise NotImplementedError("Currently only cosine distance is supported.")
        _, index = torch.topk(dist * valid_neg_mask + (1 - valid_neg_mask) * 1e6, dim=-1, k=1, largest=False)
        loss = None
        # print(camera_id_patch)
        for a_camera_id, a_feat, an_index, a_target in zip(camera_id_patch, feat_patch, index, target_patch):
            an_index = an_index[0]
            b_camera_id = camera_id_patch[an_index]
            b_feat = feat_patch[an_index]
            b_target = target_patch[an_index]
            # camera mask, 不同camera id = 1
            a_feat = F.normalize(a_feat, dim=-1)
            b_feat = F.normalize(b_feat, dim=-1)
            a_camera_mask = a_camera_id.expand(4, 4).ne(a_camera_id.expand(4, 4).t()).float()
            a_dist = -torch.matmul(a_feat, a_feat.t())
            # m1 = torch.sum(a_dist * a_camera_mask) / torch.sum(a_camera_mask)

            b_camera_mask = b_camera_id.expand(4, 4).ne(b_camera_id.expand(4, 4).t()).float()
            b_dist = -torch.matmul(b_feat, b_feat.t())
            m1_b = torch.sum(b_dist * b_camera_mask) / torch.sum(b_camera_mask)
            m1_a = torch.sum(a_dist * a_camera_mask) / torch.sum(a_camera_mask)

            if m1_b.isnan() and m1_a.isnan():
                continue
            elif m1_b.isnan():
                m1 = m1_a
            elif m1_a.isnan():
                m1 = m1_b
            else:
                m1 = (m1_b + m1_a) / 2.


            a_b_feat = torch.cat([a_feat, b_feat], dim=0)
            a_b_camera_id = torch.cat((a_camera_id, b_camera_id), dim=0)
            a_b_target = torch.cat((a_target, b_target), dim=0)
            a_b_dist = -torch.matmul(a_b_feat, a_b_feat.t())
            a_b_camera_mask = a_b_camera_id.expand(8, 8).ne(a_b_camera_id.expand(8, 8).t()).float()
            label_mask = a_b_target.expand(8, 8).ne(a_b_target.expand(8, 8).t()).float()
            a_b_mask = a_b_camera_mask * label_mask
            m2 = torch.sum(a_b_dist * a_b_mask) / torch.sum(a_b_mask)
            m1 = torch.reshape(m1, (1,1))
            m2 = torch.reshape(m2, (1,1))
            y = torch.ones_like(m2)
            if m2.isnan():
                continue
            loss_i = self.ranking_loss(m2, m1, y)
            # print(f'm2:{m2}, m1:{m1}, loss_i:{loss_i}')
            if loss is None:
                loss = loss_i
            else:
                loss = loss + loss_i
        return loss


class TripletLossCameraAware(nn.Module):
    """Triplet loss with hard positive/negative mining and camera-awareness.
    Args:
        margin (float): margin for triplet.
        distance (str): distance metric, options are ['cosine', 'euclidean'].
    """

    def __init__(self, margin=0.3, distance='cosine'):
        super(TripletLossCameraAware, self).__init__()
        if distance not in ['euclidean', 'cosine']:
            raise KeyError("Unsupported distance: {}".format(distance))
        self.distance = distance
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, input, target, camera_id):
        """
        :param input: feature matrix with shape (batch_size, feat_dim)
        :param target: ground truth labels with shape (batch_size)
        :param camera_id: camera IDs with shape (batch_size)
        :return: triplet loss value
        """
        # print('camera aware')
        n = input.size(0)

        # Compute pairwise distance
        if self.distance == 'cosine':
            input = F.normalize(input, dim=-1)
            dist = -torch.matmul(input, input.t())
        else:
            raise NotImplementedError("Currently only cosine distance is supported.")

        # Label match mask
        label_mask = target.expand(n, n).eq(target.expand(n, n).t()).float()

        # Camera ID mismatch mask (ensure samples come from different cameras)
        camera_mask = camera_id.expand(n, n).ne(camera_id.expand(n, n).t()).float()

        # Valid positive pairs: same label but different camera IDs
        valid_pos_mask = label_mask * camera_mask

        # Valid negative pairs: different labels and different camera IDs
        valid_neg_mask = (1 - label_mask) * camera_mask

        # Hardest positive
        dist_ap, _ = torch.topk(dist * valid_pos_mask - (1 - valid_pos_mask) * 1e6, dim=-1, k=1)

        # Hardest negative
        dist_an, _ = torch.topk(dist * valid_neg_mask + (1 - valid_neg_mask) * 1e6, dim=-1, k=1, largest=False)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

# class CameraAwareLoss(nn.Module):
#     """Camera-aware Triplet Loss with top-k mining.
#
#     Args:
#         margin (float): Margin for triplet loss.
#         top_k (int): The number of hardest negatives to select per anchor.
#     """
#
#     def __init__(self, margin=0.3, top_k=5):
#         super(CameraAwareLoss, self).__init__()
#         # print(margin)
#         # assert 1 < 0
#         self.margin = margin
#         self.top_k = top_k
#
#     def forward(self, input, target, camera_id):
#         """
#         :param input: feature matrix with shape (batch_size, feat_dim)
#         :param target: ground truth labels with shape (batch_size)
#         :param camera_id: camera id for each sample in the batch (shape: (batch_size))
#         :return: loss: computed loss value
#         """
#         n = input.size(0)
#         # print('camera aware:',n)
#
#         # Normalize the input features (for cosine similarity)
#         input_normalized = F.normalize(input, dim=-1)
#
#         # Compute pairwise cosine similarity
#         similarity = torch.matmul(input_normalized, input_normalized.t())  # (n, n)
#
#         # Mask to ignore same identity and same camera samples
#         target_mask = target.expand(n, n).eq(target.expand(n, n).t())  # (n, n), same identity
#         camera_mask = camera_id.expand(n, n).eq(camera_id.expand(n, n).t())  # (n, n), same camera
#
#         # We need negatives with different identities and from different cameras
#         valid_mask = ~target_mask & ~camera_mask  # (n, n), valid negative samples
#
#         # Get the top-k hardest negatives (with the highest similarity) for each anchor
#         hardest_negatives_sim = similarity * valid_mask.float()  # Masked similarity matrix
#         hardest_negatives_sim[~valid_mask] = float('-inf')  # Ensure invalid pairs are excluded
#
#         # For each anchor (i), find the top-k hardest negatives (N1, N2, ..., Nk)
#         topk_similarities, _ = torch.topk(hardest_negatives_sim, self.top_k, dim=1, largest=True)
#
#         # Compute the loss for each anchor and top-k negatives
#         loss = torch.clamp(topk_similarities, min=0)
#
#         # The final loss is the average over all anchors and their top-k negatives
#         return loss.mean()



class InfoNce(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self,
                 temperature=0.07,
                 num_instance=4):

        super(InfoNce, self).__init__()
        self.temperature = temperature
        self.ni = num_instance

    def forward(self, features):
        """
        :param features: (B, C, T)
        :param labels: (B)
        :return:
        """
        b, c, t = features.shape
        if t == 8:
            features = features.reshape(b, c, 2, 4).transpose(1, 2).reshape(b*2, c, 4)
            b, c, t = features.shape

        ni = self.ni
        features = features.reshape(b//ni, ni, c, t).permute(0, 3, 1, 2).reshape(b//ni, t*ni, c)
        features = F.normalize(features, dim=-1)
        labels = torch.arange(0, t).reshape(t, 1).repeat(1, ni).reshape(t*ni, 1)
        # (t*ni, t*ni)
        mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().cuda()  # (t*ni, t*ni)
        mask_pos = (1 - torch.eye(t*ni)).cuda()
        mask_pos = (mask * mask_pos).unsqueeze(0)

        # (b//ni, t*ni, t*ni)
        cos = torch.matmul(features, features.transpose(-1, -2))

        logits = torch.div(cos, self.temperature)
        exp_neg_logits = (logits.exp() * (1-mask)).sum(dim=-1, keepdim=True)

        log_prob = logits - torch.log(exp_neg_logits + logits.exp())
        loss = (log_prob * mask_pos).sum() / (mask_pos.sum())
        loss = - loss
        return loss


if __name__ == '__main__':
    loss = InfoNce()
    x = torch.rand(8, 16, 4).cuda()
    y = loss(x)
    print(y)
