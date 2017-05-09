# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
import pytest
import sys
from cntk import load_model
from cntk.cntk_py import DeviceKind_GPU
from cntk.device import try_set_default_device, gpu
from cntk.logging.graph import get_node_outputs
from cntk.ops.tests.ops_test_utils import cntk_device

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(abs_path)
sys.path.append(os.path.join(abs_path, "..", "..", "..", "..", "Examples", "Image", "Detection", "FastRCNN"))

from prepare_test_data import prepare_Grocery_data
from A1_GenerateInputROIs import generate_input_rois
from A2_RunCntk_py3 import train_fast_rcnn, evaluate_fast_rcnn
from A2_RunCntk import run_fastrcnn_with_config_file
from A3_ParseAndEvaluateOutput import evaluate_output
from B1_VisualizeInputROIs import generate_rois_visualization
from B2_EvaluateInputROIs import evaluate_rois
from B3_VisualizeOutputROIs import visualize_output_rois

def test_fastrcnn_grocery_visualization():
    assert generate_input_rois(testing=True)

    assert generate_rois_visualization(testing=True)

    assert evaluate_rois()

def test_fastrcnn_with_config_file(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    assert run_fastrcnn_with_config_file

def test_fastrcnn_grocery_training(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    # since we do not use a reader for evaluation we need unzipped data
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ
    grocery_path = prepare_Grocery_data()
    os.chdir(grocery_path)

    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        model_file = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
    else:
        model_file = os.path.join(abs_path, *"../../../../Examples/Image/PretrainedModels/AlexNet.model".split("/"))

    trained_model = train_fast_rcnn(model_path=model_file)
    assert evaluate_fast_rcnn(trained_model)

    assert evaluate_output()

    assert visualize_output_rois(testing=True)
