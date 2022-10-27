# --- TRAINING ---

# meta datasets and dataloader for training 
from .train.metadataset import MetaDataset
from .train.metadataloader import MetaDataLoader

# --- EVALUATION ---

# nlu dataloader for evaluation
from .val.nludataloader import NLUDataLoader

# nlu datasets for evaluation
from .val.xnlidataset import XNLIDataGenerator

NLU_TASK_DATA_GENERATOR_MAPPING = {
    "xnli": XNLIDataGenerator
}