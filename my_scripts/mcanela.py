"""
DLC 3.0
Created on Thu Jul 18 11:39:24 2024
@author: mcanela
"""

import matplotlib.pyplot as plt
import wandb

import deeplabcut
from deeplabcut.core.engine import Engine

# Specifying your project location
project = "/content/drive/MyDrive/DeepLabCut AI Residency/openfield-Pranav-2018-10-30"
config = f"{project}/config.yaml"

# Create the training dataset
deeplabcut.create_training_dataset(
    config,
    net_type="resnet_50",
    engine=Engine.PYTORCH,
)

# Training
deeplabcut.train_network(
    config,
    shuffle=1,
)
