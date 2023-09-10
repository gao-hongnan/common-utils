"""
qsub -I -l select=1:ngpus=4 -P 11003281 -l walltime=24:00:00 -q ai
"""

# from multigpu import prepare_dataloader, load_train_objs, Trainer
from transformers import (
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    AutoModel,
)
