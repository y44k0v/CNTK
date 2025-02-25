# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import os

from os import listdir
from os.path import isfile, join
from cntk import sequence, parameter, plus, reduce_sum, cntk_py
from cntk.ops.tests.ops_test_utils import cntk_device
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, FULL_DATA_SWEEP, INFINITELY_REPEAT
import cntk as C
import sys

input_dim = 69

ctf_data = '''\
0	|S0 3:1 |# <s>	|S1 3:1 |# <s>
0	|S0 4:1 |# A	|S1 32:1 |# ~AH
0	|S0 5:1 |# B	|S1 36:1 |# ~B
0	|S0 4:1 |# A	|S1 31:1 |# ~AE
0	|S0 7:1 |# D	|S1 38:1 |# ~D
0	|S0 12:1 |# I	|S1 47:1 |# ~IY
0	|S0 1:1 |# </s>	|S1 1:1 |# </s>
2	|S0 60:1 |# <s>	|S1 3:1 |# <s>
2	|S0 61:1 |# A	|S1 32:1 |# ~AH
3	|S0 60:1 |# <s>	|S1 3:1 |# <s>
3	|S0 61:1 |# A	|S1 32:1 |# ~AH
4	|S0 60:1 |# <s>	|S1 3:1 |# <s>
4	|S0 61:1 |# A	|S1 32:1 |# ~AH
5	|S0 60:1 |# <s>	|S1 3:1 |# <s>
5	|S0 61:1 |# A	|S1 32:1 |# ~AH
6	|S0 60:1 |# <s>	|S1 3:1 |# <s>
6	|S0 61:1 |# A	|S1 32:1 |# ~AH
7	|S0 60:1 |# <s>	|S1 3:1 |# <s>
7	|S0 61:1 |# A	|S1 32:1 |# ~AH
8	|S0 60:1 |# <s>	|S1 3:1 |# <s>
8	|S0 61:1 |# A	|S1 32:1 |# ~AH
9	|S0 60:1 |# <s>	|S1 3:1 |# <s>
9	|S0 61:1 |# A	|S1 32:1 |# ~AH
10	|S0 60:1 |# <s>	|S1 3:1 |# <s>
10	|S0 61:1 |# A	|S1 32:1 |# ~AH
'''


def mb_source(tmpdir, fileprefix, max_samples=FULL_DATA_SWEEP):
    ctf_file = str(tmpdir / (fileprefix + '2seqtest.txt'))
    with open(ctf_file, 'w') as f:
        f.write(ctf_data)

    mbs = MinibatchSource(CTFDeserializer(ctf_file, StreamDefs(
        features=StreamDef(field='S0', shape=input_dim, is_sparse=True),
        labels=StreamDef(field='S1', shape=input_dim, is_sparse=True)
    )),
        randomize=False, max_samples=max_samples)
    return mbs


def create_sample_model(device, writer=None):
    in1 = sequence.input_variable(shape=(input_dim,))
    labels = sequence.input_variable(shape=(input_dim,))
    p = parameter(shape=(input_dim,), init=10, device=device)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    lr_per_sample = C.learning_rate_schedule(
        [0.3, 0.2, 0.1, 0.0], C.UnitType.sample)
    learner = C.sgd(z.parameters, lr_per_sample)
    trainer = C.Trainer(z, (ce, errs), [learner], writer)
    return (trainer, in1, labels)


class MockProgressWriter(cntk_py.ProgressWriter):
    def __init__(self, expected_test_summary=None, training_summary_counter=0):
        super(MockProgressWriter, self).__init__(1, 0, 1, 0, sys.maxsize, 0)
        self.training_summary_counter = training_summary_counter
        self.test_summary_counter = 0
        self.expected_test_summary = expected_test_summary
        self.minibatch_info = []

    def on_write_training_update(self, samples, updates, aggregate_loss, aggregate_metric):
        mb_samples = samples[1] - samples[0]
        avg_loss = (aggregate_loss[1] - aggregate_loss[0]) / mb_samples
        avg_metric = (aggregate_metric[1] - aggregate_metric[0]) / mb_samples
        self.minibatch_info.append(
            (self.training_summary_counter, (avg_loss, avg_metric, mb_samples)))

    def on_write_training_summary(self, samples, updates, summaries, aggregate_loss, aggregate_metric,
                                  elapsed_milliseconds):
        self.training_summary_counter += 1

    def on_write_test_summary(self, samples, updates, summaries, aggregate_metric, elapsed_milliseconds):
        assert (self.expected_test_summary[self.test_summary_counter][
                0] == float(aggregate_metric / samples * 100.0))
        assert (self.expected_test_summary[self.test_summary_counter][1] == int(samples))
        self.test_summary_counter += 1


def test_session_sanity_check(tmpdir, device_id):
    device = cntk_device(device_id)
    t, feature, label = create_sample_model(device)
    mbs = mb_source(tmpdir, "training")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    C.training_session(
        trainer=t, mb_source=mbs,
        model_inputs_to_streams=input_map,
        mb_size=4
    ).train(device)


def test_session_max_samples(tmpdir, device_id):
    device = cntk_device(device_id)
    t, feature, label = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    C.training_session(
        trainer=t, mb_source=mbs,
        model_inputs_to_streams=input_map,
        mb_size=4, max_samples=20
    ).train(device)

    assert(t.total_number_of_samples_seen == 21)


def test_session_cross_validation_at_end(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter(expected_test_summary=[[92, 25]])
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    C.training_session(
        trainer=t, mb_source=mbs, 
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=20,
        cv_config = C.CrossValidationConfig(source=mbs1)
    ).train(device)

    assert(t.total_number_of_samples_seen == 21)
    assert(writer.test_summary_counter == 1)


def test_session_cross_validation_3_times(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter(expected_test_summary=[[92, 25], [92, 25], [92, 25]])
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    C.training_session(
        trainer=t, mb_source=mbs, 
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60,
        cv_config = C.CrossValidationConfig(source=mbs1, frequency=20, mb_size=2),
    ).train(device)

    assert(t.total_number_of_samples_seen == 61)
    assert(writer.test_summary_counter == 3)


def test_session_cross_validation_3_times_checkpoints_2_save_all(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter(expected_test_summary=[[92, 25], [92, 25], [92, 25]])
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "cv")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    test_dir = str(tmpdir)

    C.training_session(
        trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60,
        checkpoint_config = C.CheckpointConfig(frequency=35, preserve_all=True,
                                             filename=str(tmpdir / "checkpoint_save_all")),
        cv_config = C.CrossValidationConfig(source=mbs1, frequency=20)
    ).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(
        join(test_dir, f)) and f.startswith("checkpoint_save_all")]

    assert("checkpoint_save_all0" in candidates)
    assert("checkpoint_save_all0.ckp" in candidates)

    assert("checkpoint_save_all1" in candidates)
    assert("checkpoint_save_all1.ckp" in candidates)

    assert("checkpoint_save_all" in candidates)
    assert("checkpoint_save_all.ckp" in candidates)

    assert(writer.test_summary_counter == 3)


def test_session_progress_print(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter()
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    test_dir = str(tmpdir)

    C.training_session(
        trainer=t, mb_source=mbs, 
        mb_size=C.minibatch_size_schedule(4),
        model_inputs_to_streams=input_map, max_samples=60,
        progress_frequency=10
    ).train(device)

    assert(writer.training_summary_counter == 6)


def test_session_restart_from_end_checkpoint(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter()
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    test_dir = str(tmpdir)

    C.training_session(trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60, progress_frequency=20,
        checkpoint_config = C.CheckpointConfig(frequency=20,
                                             filename=str(tmpdir / "restart_from_checkpoint"))
    ).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(
        join(test_dir, f)) and f.startswith("restart_from_checkpoint")]

    assert(len(candidates) == 2)
    assert("restart_from_checkpoint" in candidates)
    assert("restart_from_checkpoint" in candidates)

    # remove information from the mock printer
    writer.minibatch_info = []
    writer.training_summary_counter = 0
    writer.testing_summary_counter = 0

    # restoring from a particular checkpoint should not cause any training
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    C.training_session(trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60, progress_frequency=20,
        checkpoint_config = C.CheckpointConfig(frequency=35, restore=True,
                                             filename=str(tmpdir / "restart_from_checkpoint"))
    ).train(device)

    assert(len(writer.minibatch_info) == 0)
    assert(writer.training_summary_counter == 0)
    assert(writer.testing_summary_counter == 0)


def test_session_restart_from_checkpoint_preserve_all(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter()
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    test_dir = str(tmpdir)

    C.training_session(trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60, progress_frequency = 20,
        checkpoint_config = C.CheckpointConfig(frequency=20, preserve_all=True,
                                             filename=str(tmpdir / "restart_from_checkpoint"))
    ).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(
        join(test_dir, f)) and f.startswith("restart_from_checkpoint")]

    assert("restart_from_checkpoint0" in candidates)
    assert("restart_from_checkpoint0.ckp" in candidates)

    assert("restart_from_checkpoint1" in candidates)
    assert("restart_from_checkpoint1.ckp" in candidates)

    assert("restart_from_checkpoint2" in candidates)
    assert("restart_from_checkpoint2.ckp" in candidates)

    assert("restart_from_checkpoint" in candidates)
    assert("restart_from_checkpoint" in candidates)

    # remove everything except for 1
    for f in candidates:
        if f != "restart_from_checkpoint1" and f != "restart_from_checkpoint1.ckp":
            os.remove(str(tmpdir / f))

    # remove information about 1 and 2 epoch from the mock printer
    first_run_minibatch_info = [i for i in writer.minibatch_info if i[0] != 0 and i[0] != 1]
    writer.minibatch_info = []
    writer.training_summary_counter = 2

    # restoring from a particular checkpoint and again save everything from the 3 epoch
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    C.training_session(
        trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60, progress_frequency=20,
        checkpoint_config = C.CheckpointConfig(frequency=20, restore=True, preserve_all= True,
                                             filename=str(tmpdir / "restart_from_checkpoint"))
    ).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(
        join(test_dir, f)) and f.startswith("restart_from_checkpoint")]

    assert("restart_from_checkpoint1" in candidates)
    assert("restart_from_checkpoint1.ckp" in candidates)

    assert("restart_from_checkpoint2" in candidates)
    assert("restart_from_checkpoint2.ckp" in candidates)

    assert("restart_from_checkpoint" in candidates)
    assert("restart_from_checkpoint.ckp" in candidates)

    assert(len(candidates) == 6)
    assert(first_run_minibatch_info == writer.minibatch_info)

    # remove everything except for 1
    for f in candidates:
        if f != "restart_from_checkpoint1" and f != "restart_from_checkpoint1.ckp":
            os.remove(str(tmpdir / f))

    # remove information about 1 and 2 epoch from the mock printer
    writer.minibatch_info = []
    writer.training_summary_counter = 2

    # renaming checkpoint 1 to generic one
    os.rename(str(tmpdir / "restart_from_checkpoint1"), str(tmpdir / "restart_from_checkpoint"))
    os.rename(str(tmpdir / "restart_from_checkpoint1.ckp"), str(tmpdir / "restart_from_checkpoint.ckp"))

    # restoring from a particular checkpoint and again save everything from the 3 epoch
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    C.training_session(
        trainer=t, mb_source=mbs,
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60, progress_frequency=20,
        checkpoint_config = C.CheckpointConfig(frequency=20, restore=True, preserve_all= True,
                                             filename=str(tmpdir / "restart_from_checkpoint"))
    ).train(device)

    candidates = [f for f in listdir(test_dir) if isfile(
        join(test_dir, f)) and f.startswith("restart_from_checkpoint")]

    assert("restart_from_checkpoint2" in candidates)
    assert("restart_from_checkpoint2.ckp" in candidates)

    assert("restart_from_checkpoint" in candidates)
    assert("restart_from_checkpoint.ckp" in candidates)

    assert(len(candidates) == 4)
    assert(first_run_minibatch_info == writer.minibatch_info)


def test_session_cv_callback_3_times(tmpdir, device_id):
    device = cntk_device(device_id)
    t, feature, label = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    counter = [0]

    def cv_callback(index, average_error, num_samples, num_mb):
        assert(counter[0] == index)
        assert average_error == 0
        assert num_samples == 0
        assert num_mb == 0
        counter[0] += 1
        return True

    C.training_session(
        trainer=t, mb_source=mbs, mb_size=4,
        model_inputs_to_streams=input_map, max_samples=60,
        cv_config = C.CrossValidationConfig(frequency=20, callback=cv_callback)
    ).train(device)
    assert counter == [3]


def test_session_cv_callback_with_cross_validation_3_times(tmpdir, device_id):
    device = cntk_device(device_id)
    t, feature, label = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    cv_mbs = mb_source(tmpdir, "cv")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    def cv_callback(index, average_error, num_samples, num_mb):
        initial_position = cv_mbs.current_position
        total_error = 0
        while True:
            mb = cv_mbs.next_minibatch(2, input_map=input_map)
            if not mb:
                break
            mb_error = t.test_minibatch(mb, device=device)
            total_error += mb_error * mb[label].num_samples

        total_samples = 25  # Please see input data
        assert((total_error * 100) / total_samples == 92)
        cv_mbs.current_position = initial_position
        return True

    C.training_session(
        trainer=t, mb_source=mbs, mb_size=4,
        model_inputs_to_streams=input_map, max_samples=60,
        cv_config = C.CrossValidationConfig(frequency=20, callback=cv_callback)
    ).train(device)

    assert(t.total_number_of_samples_seen == 61)


def test_session_cv_callback_early_exit(tmpdir, device_id):
    device = cntk_device(device_id)
    t, feature, label = create_sample_model(device)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    counter = [0]
    def cv_callback(index, average_error, num_samples, num_mb):
        assert(counter[0] == index)
        assert average_error == 0
        assert num_samples == 0
        assert num_mb == 0
        counter[0] += 1
        return counter[0] < 1

    C.training_session(
        trainer=t, mb_source=mbs, mb_size=4,
        model_inputs_to_streams=input_map,
        max_samples=60,
        cv_config = C.CrossValidationConfig(frequency=20, callback=cv_callback)
    ).train(device)
    assert counter == [1]


def test_session_with_test(tmpdir, device_id):
    device = cntk_device(device_id)
    writer = MockProgressWriter(expected_test_summary=[[92, 25]])
    t, feature, label = create_sample_model(device, writer)
    mbs = mb_source(tmpdir, "training", max_samples=INFINITELY_REPEAT)
    mbs1 = mb_source(tmpdir, "test")

    input_map = {
        feature: mbs.streams.features,
        label: mbs.streams.labels
    }

    C.training_session(
        trainer=t, mb_source=mbs, 
        mb_size=4, model_inputs_to_streams=input_map,
        max_samples=60,
        test_config = C.TestConfig(source=mbs1, mb_size=2),
    ).train(device)

    assert(t.total_number_of_samples_seen == 61)
    assert(writer.test_summary_counter == 1)

