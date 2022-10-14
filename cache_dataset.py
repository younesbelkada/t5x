import functools

from . import metrics
from . import postprocessors
from . import preprocessors

import seqio
from t5.data import get_default_vocabulary
from t5.data import postprocessors as t5_postprocessors
from t5.data import preprocessors as t5_preprocessors
from t5.evaluation import metrics as t5_metrics

MixtureRegistry = seqio.MixtureRegistry
TaskRegistry = seqio.TaskRegistry

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS
DEFAULT_EXTRA_IDS = 100
NQ_TRAIN_SPLIT_START = 7830
NQ_TRAIN_SPLIT_END = 79168
NQO_TRAIN_SPLIT_END = 79168
WQ_TRAIN_SPLIT_END = 3417
TQA_TRAIN_SPLIT_END = 78785


DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True),
    "targets": seqio.Feature(vocabulary=get_default_vocabulary(), add_eos=True)
}


# ========================== Natural Questions =================================

# Natural Questions open domain variant that most closely matches the official
# evaluation procedure.
# The model is trained to predict all ground-truth answers
# and is only considered correct if it predicts all answers for any one of the
# annotators. As in the official evaluation, we consider questions with fewer
# than two non-null annotations unanswerable (given the context) but because we
# cannot predict unanswerability without the context, we only compute the recall
# metric. Further, because our model does not have access to the oracle context,
# we also normalize predicted and ground-truth answers when comparing them.

# This task uses a portion of the train set for validation.
TaskRegistry.add(
    "natural_questions_nocontext",
    source=seqio.TfdsDataSource(
        tfds_name="natural_questions:0.0.2",
        splits={
            "train": f"train[{NQ_TRAIN_SPLIT_START}:{NQ_TRAIN_SPLIT_END}]",
            "validation": f"train[:{NQ_TRAIN_SPLIT_START}]",
            "test": "validation"
        }),
    preprocessors=[
        preprocessors.natural_questions_nocontext,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocessors.natural_questions,
    metric_fns=[
        functools.partial(
            metrics.natural_questions,
            # Train set does not contain multiple annotations.
            non_null_threshold=1)
    ])
# This task uses full train split and reports metrics on the NQ validation split
# (which is the test set in the open domain setting).
TaskRegistry.add(
    "natural_questions_nocontext_test",
    source=seqio.TfdsDataSource(tfds_name="natural_questions:0.0.2"),
    preprocessors=[
        preprocessors.natural_questions_nocontext,
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
    postprocess_fn=postprocessors.natural_questions,
    metric_fns=[metrics.natural_questions])