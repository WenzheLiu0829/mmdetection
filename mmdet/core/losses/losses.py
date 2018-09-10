import torch
import torch.nn.functional as F

from ..bbox_ops import bbox_transform_inv, bbox_overlaps


def weighted_cross_entropy(pred, label, weight, ave_factor=None):
    if ave_factor is None:
        ave_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.cross_entropy(pred, label, size_average=False, reduce=False)
    return torch.sum(raw * weight)[None] / ave_factor


def weighted_nll_loss(pred, label, weight, ave_factor=None):
    if ave_factor is None:
        ave_factor = max(torch.sum(weight > 0).float().item(), 1.)
    raw = F.nll_loss(pred, label, size_average=False, reduce=False)
    return torch.sum(raw * weight)[None] / ave_factor


def weighted_binary_cross_entropy(pred, label, weight, ave_factor=None):
    if ave_factor is None:
        ave_factor = max(torch.sum(weight > 0).float().item(), 1.)
    return F.binary_cross_entropy_with_logits(
        pred, label.float(), weight.float(),
        size_average=False)[None] / ave_factor


def smooth_l1_loss(pred, target, beta=1.0, size_average=True, reduce=True):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    if size_average:
        loss /= pred.numel()
    if reduce:
        loss = loss.sum()
    return loss


def weighted_smoothl1(pred, target, weight, beta=1.0, ave_factor=None):
    if ave_factor is None:
        ave_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, size_average=False, reduce=False)
    return torch.sum(loss * weight)[None] / ave_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       size_average=True):
    pred_sigmoid = pred.sigmoid()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    return F.binary_cross_entropy_with_logits(
        pred, target, weight, size_average=size_average)


def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                ave_factor=None,
                                num_classes=80):
    if ave_factor is None:
        ave_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        size_average=False)[None] / ave_factor


def mask_cross_entropy(pred, target, label):
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, size_average=True)[None]


def weighted_mask_cross_entropy(pred, target, weight, label):
    num_rois = pred.size()[0]
    num_samples = torch.sum(weight > 0).float().item() + 1e-6
    assert num_samples >= 1
    inds = torch.arange(0, num_rois).long().cuda()
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight, size_average=False)[None] / num_samples


def weighted_iou_loss(deltas, rois, gts, weights, ave_factor=None):
    inds = torch.nonzero(weights[:, 0] > 0)
    if ave_factor is None:
        ave_factor = inds.numel() + 1e-6
    if inds.numel() > 0:
        inds = inds.squeeze(1)
    else:
        return (deltas * weights[:, :2]).sum()[None] / ave_factor
    denorm_deltas = deltas[inds, :]
    rois_ = rois[inds, :]
    gts_ = gts[inds, :]
    dw = denorm_deltas[:, 0::2]
    dh = denorm_deltas[:, 1::2]
    gx = ((rois_[:, 0] + rois_[:, 2]) * 0.5).unsqueeze(1).expand_as(dw)
    gy = ((rois_[:, 1] + rois_[:, 3]) * 0.5).unsqueeze(1).expand_as(dh)
    pw = (rois_[:, 2] - rois_[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois_[:, 3] - rois_[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(rois_)

    lt = torch.max(bboxes[:, :2], gts_[:, :2])
    rb = torch.min(bboxes[:, 2:], gts_[:, 2:])

    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (bboxes[:, 2] - bboxes[:, 0] + 1) * (
        bboxes[:, 3] - bboxes[:, 1] + 1)
    area2 = (gts_[:, 2] - gts_[:, 0] + 1) * (gts_[:, 3] - gts_[:, 1] + 1)
    ious = overlap / (area1 + area2 - overlap)
    diff = 1.0 - ious

    return diff.sum()[None] / ave_factor


def bounded_iou_loss(pred,
                     target_means,
                     target_stds,
                     rois,
                     gts,
                     weights,
                     beta=0.2,
                     ave_factor=None,
                     eps=1e-3):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164
    """

    inds = torch.nonzero(weights[:, 0] > 0)
    if ave_factor is None:
        ave_factor = inds.numel() + 1e-6

    if inds.numel() > 0:
        inds = inds.squeeze(1)
    else:
        return (pred[:, 2:] * weights[:, :2]).sum()[None] / ave_factor

    pred_ = pred[inds, :]
    rois_ = rois[inds, :]
    gts_ = gts[inds, :]
    # valid_pred = pred_.masked_select(weights[inds, :].byte()).view(-1, 4)
    pred_bboxes = bbox_transform_inv(
        rois_, pred_, target_means, target_stds, wh_ratio_clip=1e-6)
    pred_ctrx = (pred_bboxes[:, 0] + pred_bboxes[:, 2]) * 0.5
    pred_ctry = (pred_bboxes[:, 1] + pred_bboxes[:, 3]) * 0.5
    pred_w = pred_bboxes[:, 2] - pred_bboxes[:, 0] + 1
    pred_h = pred_bboxes[:, 3] - pred_bboxes[:, 1] + 1
    with torch.no_grad():
        gt_ctrx = (gts_[:, 0] + gts_[:, 2]) * 0.5
        gt_ctry = (gts_[:, 1] + gts_[:, 3]) * 0.5
        gt_w = gts_[:, 2] - gts_[:, 0] + 1
        gt_h = gts_[:, 3] - gts_[:, 1] + 1

    dx = gt_ctrx - pred_ctrx
    dy = gt_ctry - pred_ctry

    loss_dx = 1 - torch.max((gt_w - 2 * dx.abs()) /
                            (gt_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max((gt_h - 2 * dy.abs()) /
                            (gt_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(gt_w / (pred_w + eps), pred_w / (gt_w + eps))
    loss_dh = 1 - torch.min(gt_h / (pred_h + eps), pred_h / (gt_h + eps))
    loss_comb = torch.stack(
        [loss_dx, loss_dy, loss_dw, loss_dh], dim=-1).view(
            loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    loss = loss.sum()[None] / ave_factor
    return loss


def rank_matching_loss(proposals_list,
                       gt_bboxes_list,
                       gt_labels_list,
                       assigned_gt_inds_list,
                       cls_score,
                       bbox_pred,
                       target_means,
                       target_stds,
                       ave_factor=None):

    losses = []
    start = 0
    count = 0
    for img_id in range(len(proposals_list)):
        end = start + proposals_list[img_id].size(0)
        bbox_deltas = bbox_pred.detach()[start:end]
        cls_scores_select = cls_score[start:end]
        proposals = proposals_list[img_id]
        gt_bboxes = gt_bboxes_list[img_id]
        gt_labels = gt_labels_list[img_id]
        assigned_gt_inds = assigned_gt_inds_list[img_id]

        pred_bboxes = bbox_transform_inv(
            proposals,
            bbox_deltas,
            target_means,
            target_stds,
            wh_ratio_clip=1e-6)
        overlaps = bbox_overlaps(pred_bboxes, gt_bboxes, is_aligned=True)
        assert cls_scores_select.size(0) == proposals.size(0)
        softmax_cls_score = F.softmax(cls_scores_select, dim=1)
        scores = softmax_cls_score[torch.arange(proposals.size(0)).long().cuda(
        ), gt_labels.long()]
        pred_labels = softmax_cls_score.detach().argmax(dim=1)

        for i in torch.unique(assigned_gt_inds.cpu()):
            inds = assigned_gt_inds == i.item()
            overlaps_ = overlaps[inds]
            scores_ = scores[inds]
            overlaps_diff = overlaps_[:, None] - overlaps_
            iou_matrix = overlaps_diff
            # iou_matrix = ((overlaps_[:, None] - overlaps_) > 0).float() - (
            #     (overlaps_[:, None] - overlaps_) < 0).float()
            score_matrix = scores_[:, None] - scores_

            loss_matrix = F.relu(-score_matrix * iou_matrix)

            beta = 0.05
            loss = torch.where(loss_matrix < beta,
                               0.5 * loss_matrix * loss_matrix / beta,
                               loss_matrix - 0.5 * beta)
            if overlaps_.size(0) > 1:
                count += loss_matrix.size(0) * 2
            losses.append(loss)
        start = end
    assert end == cls_score.size(0)
    loss = sum([loss_.sum() for loss_ in losses])
    if ave_factor is None:
        ave_factor = max(1, count)
    loss = loss.sum()[None] / ave_factor

    return loss


def accuracy(pred, target, topk=1):
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True

    maxk = max(topk)
    _, pred_label = pred.topk(maxk, 1, True, True)
    pred_label = pred_label.t()
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / pred.size(0)))
    return res[0] if return_single else res
