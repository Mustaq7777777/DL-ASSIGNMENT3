# sweep_utils.py
# This file contains utilities for hyperparameter tuning with Weights & Biases

import os
import wandb
import torch
import random
import numpy as np

def seed_everything(seed=42):
    """
    Set random seed for all major libraries.
    
    This ensures reproducibility across runs.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def setup_sweep_config(with_attention=True):
    """
    Create a configuration for W&B hyperparameter sweep.
    
    Args:
        with_attention (bool): Whether to run sweep for attention model
        
    Returns:
        dict: Sweep configuration
    """
    sweep_name = 'Transliteration_with_Attention' if with_attention else 'Transliteration_without_Attention'
    
    # Define sweep configuration
    sweep_cfg = {
        'method': 'bayes',  # Use Bayesian optimization
        'name': sweep_name,
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            # Model architecture
            'emb_size': {'values': [128, 256, 512]},
            'hidden_size': {'values': [128, 256, 512, 1024]},
            'enc_layers': {'values': [1, 2, 3, 4]},
            'cell': {'values': ['RNN', 'GRU', 'LSTM']},  
            'bidirectional': {'values': [True, False]},  # Bidirectional encode
            
            # Training parameters
            'dropout': {'values': [0.0, 0.1, 0.2, 0.3, 0.5]},
            'lr': {'values': [1e-4, 2e-4, 5e-4, 8e-4, 1e-3]},
            'batch_size': {'values': [32, 64, 128]},
            'epochs': {'values': [10, 15, 20]},
            'teacher_forcing': {'values': [0.3, 0.5, 0.7, 1.0]},  # Explicit teacher forcing
            'optimizer': {'values': ['Adam', 'NAdam']},  # Added optimizer options
            # Reproducibility
            'seed': {'values': [42, 43, 44, 45, 46]},  # Different seeds for robustness
        }
    }
    
    return sweep_cfg

def run_sweep_agent(sweep_id, func, count=20):
    """
    Run a Weights & Biases sweep agent.
    
    Args:
        sweep_id (str): ID of the sweep
        func (callable): Objective function for the sweep
        count (int): Maximum number of runs
    """
    wandb.agent(sweep_id, function=func, count=count)

def init_sweep(with_attention=True, entity=None, project='DL_assignment_3'):
    """
    Initialize a Weights & Biases sweep.
    
    Args:
        with_attention (bool): Whether to run sweep for attention model
        entity (str): W&B entity (username or team name)
        project (str): W&B project name
        
    Returns:
        str: Sweep ID
    """
    sweep_config = setup_sweep_config(with_attention)
    sweep_id = wandb.sweep(sweep_config, entity=entity, project=project)
    return sweep_id

def create_sweep_objective(train_func, data_tensors, data_vocab, language='tel'):
    """
    Create an objective function for W&B sweep.
    
    Args:
        train_func (callable): Training function
        data_tensors (tuple): Tensors for training, validation, and testing
        data_vocab (tuple): Vocabularies
        language (str): Language code
        
    Returns:
        callable: Objective function for sweep
    """
    def objective():
        # Initialize wandb run
        run = wandb.init()
        config = run.config
        
        # Set seed for reproducibility
        seed_everything(config.seed)
        
        # Unpack data
        (eng_matrix_train, tel_matrix_train, 
         eng_matrix_valid, tel_matrix_valid, 
         eng_matrix_test, tel_matrix_test) = data_tensors
        
        eng_vocab, tel_vocab = data_vocab
        
        # Train model with swept parameters
        train_func(
            config.cell,
            config.bidirectional,
            config.emb_size,
            config.dropout,
            config.dropout,
            config.enc_layers,
            config.enc_layers,
            config.hidden_size,
            config.batch_size,
            with_attention,
            config.lr,
            config.epochs,
            eng_matrix_train, tel_matrix_train,
            eng_matrix_valid, tel_matrix_valid,
            eng_matrix_test, tel_matrix_test,
            eng_vocab, tel_vocab,
            language,
            True  # use_wandb
        )
    
    return objective