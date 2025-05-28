from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import numpy as np
import argparse
import torch
import sys
import cv2
import os


from hybrid_model import *
from torch_loader import EuroSATDataModule


if __name__=='__main__':
     # Train Model
    torch.multiprocessing.set_start_method('spawn')  
    torch.set_float32_matmul_precision('high')
    # Instantiate LightningModule and DataModule

    network = LightningAlexNetModule(n_qubits=4, num_classes=10)
    
    classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture','PermanentCrop', 'Residential', 'River', 'SeaLake']

    data_module = EuroSATDataModule(
        whitelist_classes = classes,
        batch_size        = 16, 
        bands             = [3,2,1],
        num_workers       = 4,
    )
    
    log_name = 'hybrid_model'

    tb_logger   = pl.loggers.TensorBoardLogger(os.path.join('lightning_logs','classifiers'), name=log_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('saved_models','classifiers'),
        filename=log_name,
        monitor='val_loss',
        save_top_k=1,
        mode='min',
    )

    # Instantiate Trainer
    trainer = pl.Trainer(max_epochs=30, callbacks=[checkpoint_callback], logger=tb_logger)

    # Train the model
    trainer.fit(network, data_module)
