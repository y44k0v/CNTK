import os, sys
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))

import pytest
import numpy as np
from cntk import input as input_variable, user_function
from rpn.proposal_layer import ProposalLayer as CntkProposalLayer
from rpn.proposal_target_layer import ProposalTargetLayer as CntkProposalTargetLayer
from caffe_layers.proposal_layer_caffe import ProposalLayer as CaffeProposalLayer
from caffe_layers.proposal_target_layer_caffe import ProposalTargetLayer as CaffeProposalTargetLayer

def test_proposal_layer():
    cls_prob_shape_cntk = (2,9,61,61)
    cls_prob_shape_caffe = (18,61,61)
    rpn_bbox_shape = (36, 61, 61)
    im_info = [1000, 1000, 1]
    test_specific_values = False

    # Create input tensors with values
    if test_specific_values:
        bg_probs = [0.2, 0.05, 0.05, 0.0, 0.0, 0.1, 0.1, 0.0, 0.5]
        fg_probs = np.ones(9) - bg_probs
        cls_prob = np.zeros((61, 61, 9, 2))
        cls_prob[:, :, :, 0] = bg_probs
        cls_prob[:, :, :, 1] = fg_probs
        cls_prob = np.ascontiguousarray(cls_prob.transpose(3, 2, 1, 0)).astype(np.float32)

        bbox_pred = [0.2, -0.1, 0.3, -0.4] * 9
        rpn_bbox_pred = np.zeros((61, 61, 36))
        rpn_bbox_pred[:, :, :] = bbox_pred
        rpn_bbox_pred = np.ascontiguousarray(rpn_bbox_pred.transpose(2, 1, 0)).astype(np.float32)
    else:
        cls_prob =  np.random.random_sample(cls_prob_shape_cntk).astype(np.float32)
        rpn_bbox_pred = np.random.random_sample(rpn_bbox_shape).astype(np.float32)

    # Create CNTK layer and call forward
    cls_prob_var = input_variable(cls_prob_shape_cntk)
    rpn_bbox_var = input_variable(rpn_bbox_shape)

    cntk_layer = user_function(CntkProposalLayer(cls_prob_var, rpn_bbox_var, im_info=im_info))
    state, cntk_output = cntk_layer.forward({cls_prob_var: [cls_prob], rpn_bbox_var: [rpn_bbox_pred]})
    cntk_proposals = cntk_output[next(iter(cntk_output))][0]

    # Create Caffe layer and call forward
    cls_prob_caffe = cls_prob.reshape(cls_prob_shape_caffe)
    bottom = [np.array([cls_prob_caffe]),np.array([rpn_bbox_pred]),np.array([im_info])]
    top = None # handled through return statement in caffe layer for unit testing

    param_str = "'feat_stride': 16"
    caffe_layer = CaffeProposalLayer()
    caffe_layer.set_param_str(param_str)
    caffe_layer.setup(bottom, top)
    caffe_output = caffe_layer.forward(bottom, top)
    caffe_proposals = caffe_output[:,1:]

    # assert that results are exactly the same
    assert cntk_proposals.shape == caffe_proposals.shape
    assert np.allclose(cntk_proposals, caffe_proposals, rtol=0.0, atol=0.0)
    print("Verified ProposalLayer")

def test_proposal_target_layer():
    all_rois_shape_cntk = (400,4)
    num_gt_boxes = 50
    gt_boxes_shape_cntk = (50,5)
    im_info = [1000, 1000, 1]

    # Create input tensors with values
    all_rois = np.random.random_sample(all_rois_shape_cntk).astype(np.float32)

    x1y1 = np.random.random_sample((num_gt_boxes, 2)) * 500
    wh = np.random.random_sample((num_gt_boxes, 2)) * 400
    x2y2 = x1y1 + wh + 50
    label = np.random.random_sample((num_gt_boxes, 1))
    label = (label * 17.0)
    gt_boxes = np.hstack((x1y1, x2y2, label)).astype(np.float32)

    # Create CNTK layer and call forward
    all_rois_var = input_variable(all_rois_shape_cntk)
    gt_boxes_var = input_variable(gt_boxes_shape_cntk)

    cntk_layer = user_function(CntkProposalTargetLayer(all_rois_var, gt_boxes_var, num_classes=17, im_info=im_info, deterministic=True))
    state, cntk_output = cntk_layer.forward({all_rois_var: [all_rois], gt_boxes_var: [gt_boxes]})

    roi_key = [k for k in cntk_output if 'ptl_roi' in str(k)][0]
    labels_key = [k for k in cntk_output if 'ptl_labels' in str(k)][0]
    bbox_key = [k for k in cntk_output if 'ptl_bbox' in str(k)][0]
    bbox_w_key = [k for k in cntk_output if 'ptl_bbox_w' in str(k)][0]

    cntk_rois = cntk_output[roi_key][0]
    cntk_labels_one_hot = cntk_output[labels_key][0]
    cntk_bbox_targets = cntk_output[bbox_key][0]
    cntk_bbox_inside_weights = cntk_output[bbox_w_key][0]

    cntk_labels = np.argmax(cntk_labels_one_hot, axis=1)

    # Create Caffe layer and call forward
    zeros = np.zeros((all_rois.shape[0], 1), dtype=gt_boxes.dtype)
    all_rois_caffe = np.hstack((zeros, all_rois))

    bottom = [np.array(all_rois_caffe),np.array(gt_boxes)]
    top = None # handled through return statement in caffe layer for unit testing

    param_str = "'num_classes': 17"
    caffe_layer = CaffeProposalTargetLayer()
    caffe_layer.set_param_str(param_str)
    caffe_layer.setup(bottom, top)
    caffe_layer.set_deterministic_mode()

    caffe_rois, caffe_labels, caffe_bbox_targets, caffe_bbox_inside_weights = caffe_layer.forward(bottom, top)
    caffe_rois = caffe_rois[:,1:]

    num_caffe_rois = caffe_rois.shape[0]
    cntk_rois = cntk_rois[:num_caffe_rois,:]
    cntk_labels = cntk_labels[:num_caffe_rois]
    cntk_bbox_targets = cntk_bbox_targets[:num_caffe_rois,:]
    cntk_bbox_inside_weights = cntk_bbox_inside_weights[:num_caffe_rois,:]

    # assert that results are exactly the same
    assert cntk_rois.shape == caffe_rois.shape
    assert cntk_labels.shape == caffe_labels.shape
    assert cntk_bbox_targets.shape == caffe_bbox_targets.shape
    assert cntk_bbox_inside_weights.shape == caffe_bbox_inside_weights.shape
    assert np.allclose(cntk_rois, caffe_rois, rtol=0.0, atol=0.0)
    print("Verified ProposalTargetLayer")

if __name__ == '__main__':
    test_proposal_layer()
    test_proposal_target_layer()
