import os
import torch
import wandb

from models.maskrcnn import MaskRCNNModel
from datasets.tdd import TuftsDentalDataModule

ROOT_DIR = "./data"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
STEP_SIZE = 5
GAMMA = 0.1
N_EPOCHS = 10

# Initialize W&B project
wandb.init(
    project="mask-rcnn-tufts-dental",
    # track hyperparameters and run metadata
    config={
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
        "n_epochs": N_EPOCHS,
    },
)

# Set the device
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# elif torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the paths
image_dir = os.path.join(ROOT_DIR, "Radiographs")
annotations_path = os.path.join(ROOT_DIR, "Segmentation", "teeth_polygon.json")

# DataModule
tdd_module = TuftsDentalDataModule(
    annotations_path=annotations_path,
    image_dir=image_dir,
    batch_size=BATCH_SIZE,
)
tdd_module.setup()

# Model
num_classes = 33
model = MaskRCNNModel(
    num_classes=num_classes,
    learning_rate=LEARNING_RATE,
    step_size=STEP_SIZE,
    gamma=GAMMA,
)


# Train
model.train_model(
    train_loader=tdd_module.train_dataloader(),
    val_loader=tdd_module.val_dataloader(),
    device=device,
    num_epochs=N_EPOCHS,
)
