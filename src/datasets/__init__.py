# --- TRAINING ---

# meta datasets and dataloader for training
from .train.metadataset import MetaDataset
from .train.metadataloader import MetaDataLoader

# --- EVALUATION ---

# nlu dataloader for evaluation
from .val.nludataloader import NLUDataLoader

# nlu datasets for evaluation
from .val.nludataset import NLUDataset
from .val.xnlidataset import (
    XNLIStandardDataGenerator,
    XNLIFewShotDataGenerator,
    XNLICrossLingualDataGenerator,
)

NLU_STANDARD_TASK_DATA_GENERATOR_MAPPING = {"xnli": XNLIStandardDataGenerator}

NLU_FEW_SHOT_TASK_DATA_GENERATOR_MAPPING = {"xnli": XNLIFewShotDataGenerator}

NLU_CROSS_LINGUAL_TASK_DATA_GENERATOR_MAPPING = {"xnli": XNLICrossLingualDataGenerator}
