import mmcv
import torch

from .geometry import bbox_overlaps
from .sampling import bbox_assign, bbox_sampling
from .transforms import bbox_transform, bbox_transform_inv


def bbox_target(pos_proposals_list,
                    neg_proposals_list,
                    pos_gt_bboxes_list,
                    pos_gt_labels_list,
                    target_means,
                    target_stds,
                    cfg,
                    reg_num_classes=1,
                    debug_imgs=None,
                    return_list=False):
    img_per_gpu = len(pos_proposals_list)
    all_labels = []
    all_label_weights = []
    all_bbox_targets = []
    all_bbox_weights = []
    for img_id in range(img_per_gpu):
        pos_proposals = pos_proposals_list[img_id]
        neg_proposals = neg_proposals_list[img_id]
        pos_gt_bboxes = pos_gt_bboxes_list[img_id]
        pos_gt_labels = pos_gt_labels_list[img_id]
        debug_img = debug_imgs[img_id] if cfg.debug else None
        labels, label_weights, bbox_targets, bbox_weights = proposal_target_single(
            pos_proposals, neg_proposals, pos_gt_bboxes, pos_gt_labels,
            target_means, target_stds, reg_num_classes, cfg, debug_img)
        all_labels.append(labels)
        all_label_weights.append(label_weights)
        all_bbox_targets.append(bbox_targets)
        all_bbox_weights.append(bbox_weights)

    if return_list:
        return all_labels, all_label_weights, all_bbox_targets, all_bbox_weights

    labels = torch.cat(all_labels, 0)
    label_weights = torch.cat(all_label_weights, 0)
    bbox_targets = torch.cat(all_bbox_targets, 0)
    bbox_weights = torch.cat(all_bbox_weights, 0)
    return labels, label_weights, bbox_targets, bbox_weights


def proposal_target_single(pos_proposals,
                           neg_proposals,
                           pos_gt_bboxes,
                           pos_gt_labels,
                           target_means,
                           target_stds,
                           reg_num_classes,
                           cfg,
                           debug_img=None):
    num_pos = pos_proposals.size(0)
    num_neg = neg_proposals.size(0)
    num_samples = num_pos + num_neg
    labels = pos_proposals.new_zeros(num_samples, dtype=torch.long)
    label_weights = pos_proposals.new_zeros(num_samples)
    bbox_targets = pos_proposals.new_zeros(num_samples, 4)
    bbox_weights = pos_proposals.new_zeros(num_samples, 4)
    if num_pos > 0:
        labels[:num_pos] = pos_gt_labels
        pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
        label_weights[:num_pos] = pos_weight
        pos_bbox_targets = bbox_transform(pos_proposals, pos_gt_bboxes,
                                          target_means, target_stds)
        bbox_targets[:num_pos, :] = pos_bbox_targets
        bbox_weights[:num_pos, :] = 1
    if num_neg > 0:
        label_weights[-num_neg:] = 1.0
    if reg_num_classes > 1:
        bbox_targets, bbox_weights = expand_target(bbox_targets, bbox_weights,
                                                   labels, reg_num_classes)
    if cfg.debug:
        restore_bboxes = bbox_transform_inv(pos_proposals, pos_bbox_targets,
                                            target_means, target_stds)
        print('target shape', restore_bboxes.shape)
        cvb.draw_bboxes(
            debug_img.copy(),
            restore_bboxes.cpu().numpy(),
            win_name='roi target')
    return labels, label_weights, bbox_targets, bbox_weights


def sample_proposals(proposals_list, gt_bboxes_list, gt_crowds_list,
                     gt_labels_list, cfg):
    cfg_list = [cfg for _ in range(len(proposals_list))]
    results = map(sample_proposals_single, proposals_list, gt_bboxes_list,
                  gt_crowds_list, gt_labels_list, cfg_list)
    # list of tuple to tuple of list
    return tuple(map(list, zip(*results)))


def sample_proposals_single(proposals,
                            gt_bboxes,
                            gt_crowds,
                            gt_labels,
                            cfg,
                            debug_img=None):
    proposals = proposals[:, :4]
    assigned_gt_inds, assigned_labels, argmax_overlaps, max_overlaps = \
        bbox_assign(
            proposals, gt_bboxes, gt_crowds, gt_labels, cfg.pos_iou_thr,
            cfg.neg_iou_thr, cfg.pos_iou_thr, cfg.crowd_thr)
    if cfg.add_gt_as_proposals:
        proposals = torch.cat([gt_bboxes, proposals], dim=0)
        gt_assign_self = torch.arange(
            1, len(gt_labels) + 1, dtype=torch.long, device=proposals.device)
        assigned_gt_inds = torch.cat([gt_assign_self, assigned_gt_inds])
        assigned_labels = torch.cat([gt_labels, assigned_labels])

    pos_inds, neg_inds = bbox_sampling(
        assigned_gt_inds, cfg.roi_batch_size, cfg.pos_fraction, cfg.neg_pos_ub,
        cfg.pos_balance_sampling, max_overlaps, cfg.neg_balance_thr)
    pos_proposals = proposals[pos_inds]
    neg_proposals = proposals[neg_inds]
    pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1
    pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
    pos_gt_labels = assigned_labels[pos_inds]
    if cfg.debug:
        print('----------------Proposal sampling debug info----------------')
        print('after sampling, pos: {}, neg: {}, ignore: {}'.format(
            pos_inds.numel(),
            neg_inds.numel(),
            assigned_gt_inds.numel() - pos_inds.numel() - neg_inds.numel()))
        neg_max_gt = gt_bboxes[argmax_overlaps[neg_inds], :]
        neg_iou = torch.diag(bbox_overlaps(neg_proposals, neg_max_gt))
        pos_iou = torch.diag(bbox_overlaps(pos_proposals, pos_gt_bboxes))
        print('pos iou', pos_iou.cpu().numpy())
        print('pos_labels', pos_gt_labels.cpu().numpy())
        print('neg iou', neg_iou.cpu().numpy())
        if cfg.neg_balance_thr > 0:
            print('easy neg num: {}, hard neg num: {}'.format(
                torch.sum(neg_iou < cfg.neg_balance_thr),
                torch.sum(neg_iou >= cfg.neg_balance_thr)))
        cvb.draw_bboxes(
            debug_img.copy(),
            [gt_bboxes.cpu().numpy(),
             pos_proposals.cpu().numpy()], [cvb.Color.green, cvb.Color.blue],
            win_name='pos rois',
            wait_time=10)
        cvb.draw_bboxes(
            debug_img.copy(),
            [gt_bboxes.cpu().numpy(),
             neg_proposals.cpu().numpy()], [cvb.Color.green, cvb.Color.red],
            win_name='neg rois',
            wait_time=10)
    return (pos_inds, neg_inds, pos_proposals, neg_proposals,
            pos_assigned_gt_inds, pos_gt_bboxes, pos_gt_labels)


def expand_target(bbox_targets, bbox_weights, labels, num_classes):
    bbox_targets_expand = bbox_targets.new_zeros((bbox_targets.size(0),
                                                  4 * num_classes))
    bbox_weights_expand = bbox_weights.new_zeros((bbox_weights.size(0),
                                                  4 * num_classes))
    for i in torch.nonzero(labels > 0).squeeze(-1):
        start, end = labels[i] * 4, (labels[i] + 1) * 4
        bbox_targets_expand[i, start:end] = bbox_targets[i, :]
        bbox_weights_expand[i, start:end] = bbox_weights[i, :]
    return bbox_targets_expand, bbox_weights_expand
