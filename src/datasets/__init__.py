# --- TRAINING ---

# meta datasets and dataloader for training
from .train.metadataset import MetaDataset
from .train.metadataloader import MetaDataLoader

# --- EVALUATION ---

# nlu datasets for evaluation
from .val.nlutask import NLUTaskGenerator
from .val.xnli import (
    XNLIGenerator,
)
from .val.mlqa import (
    MLQAGenerator,
)

NLU_TASK_GENERATOR_MAP = {
    "xnli": XNLIGenerator,
    "mlqa": MLQAGenerator,
}
