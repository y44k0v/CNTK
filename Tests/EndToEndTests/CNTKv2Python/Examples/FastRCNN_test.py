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
from A2_RunWithBSModel import run_fastrcnn_with_config_file
from A3_ParseAndEvaluateOutput import evaluate_output
from B1_VisualizeInputROIs import generate_rois_visualization
from B2_EvaluateInputROIs import evaluate_rois
from B3_VisualizeOutputROIs import visualize_output_rois

grocery_path = prepare_Grocery_data()

python34_only = pytest.mark.skipif(sys.version_info[:2] ! (3,4), reason="requires python 3.4")
linux_only = pytest.mark.skipif(sys.platform == 'win32', reason="it runs currently only in linux")

@python3_only
@linux_only
def test_fastrcnn_grocery_visualization():
    assert generate_input_rois(testing=True)

    assert generate_rois_visualization(testing=True)

    assert evaluate_rois()

@python3_only
@linux_only
def test_fastrcnn_with_config_file(device_id):
    assert generate_input_rois(testing=True)

    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    assert run_fastrcnn_with_config_file()

@python3_only
@linux_only
def test_fastrcnn_grocery_training(device_id):
    if cntk_device(device_id).type() != DeviceKind_GPU:
        pytest.skip('test only runs on GPU') # it runs very slow in CPU
    try_set_default_device(cntk_device(device_id))

    assert generate_input_rois(testing=True)

    # since we do not use a reader for evaluation we need unzipped data
    externalData = 'CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY' in os.environ

    if externalData:
        extPath = os.environ['CNTK_EXTERNAL_TESTDATA_SOURCE_DIRECTORY']
        model_file = os.path.join(extPath, "PreTrainedModels", "AlexNet", "v0", "AlexNet.model")
    else:
        model_file = os.path.join(abs_path, *"../../../../Examples/Image/PretrainedModels/AlexNet.model".split("/"))

    from A2_RunWithPyModel import train_fast_rcnn, evaluate_fast_rcnn

    trained_model = train_fast_rcnn(model_path=model_file)

    assert evaluate_fast_rcnn(trained_model)

    assert evaluate_output()

    assert visualize_output_rois(testing=True)
