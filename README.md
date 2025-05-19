# Neural Transliteration with Sequence-to-Sequence Models

This repository contains a PyTorch implementation of sequence-to-sequence neural models for transliteration tasks. The models are trained on the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), which includes paired Latin-script and native-script text for 12 South Asian languages.

![Transliteration Example](https://img.shields.io/badge/Example-ramesh%20%E2%86%92%20%E0%B0%B0%E0%B0%AE%E0%B1%87%E0%B0%B6%E0%B1%8D-blue)
![Languages](https://img.shields.io/badge/Languages-12%20South%20Asian-green)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red)

## Overview

The transliteration system converts words written in Latin script (English) to their corresponding representation in a target language script (e.g., Telugu, Hindi, Tamil). It uses sequence-to-sequence (seq2seq) neural networks with:

- Encoder-decoder architecture
- Multiple RNN cell types (RNN, GRU, LSTM)
- Optional bidirectional encoder
- Optional attention mechanism
- Teacher forcing during training

## Features

-  Supports all 12 languages from the Dakshina dataset
-  Multiple RNN cell types (RNN, GRU, LSTM)
-  Bidirectional encoder option
-  Attention mechanism option
-  Hyperparameter tuning with Weights & Biases
-  Detailed training metrics and visualizations
-  Evaluation with exact match accuracy

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/neural-transliteration.git
cd neural-transliteration
pip install -r requirements.txt
```

### Requirements

- Python 3.6+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm
- wandb (optional, for hyperparameter tuning)

## Dataset

This project uses the [Dakshina dataset](https://github.com/google-research-datasets/dakshina), which includes paired Latin-script and native-script text for 12 South Asian languages. The dataset will be automatically downloaded and extracted when you run the training script for the first time.

## Files Structure

- `models.py`: Neural network model definitions
- `train.py`: Main script for training
- `train_utils.py`: Training and evaluation utilities
- `data_utils.py`: Data loading and processing functions
- `sweep_utils.py`: Hyperparameter tuning utilities

## Usage

### Basic Training

To train a model with default parameters:

```bash
python train.py
```

This will train a GRU-based model with attention for Telugu transliteration.

### Customizing Model Parameters

You can customize various aspects of the model using command-line arguments:

```bash
python train.py -lang tel -ct GRU -bi True -attn True -emd_size 256 -h_size 512 -epoch 15
```

### Command-Line Arguments

#### Dataset Parameters

| Argument | Description | Default |
| --- | --- | --- |
| `-lang`, `--language` | Language code (e.g., tel for Telugu) | `tel` |
| `-o`, `--output_dir` | Directory to save outputs | `outputs` |
| `--base_path` | Base path to the Dakshina dataset | `None` |

#### Model Architecture Parameters

| Argument | Description | Default |
| --- | --- | --- |
| `-ct`, `--cell_type` | RNN cell type (RNN, GRU, or LSTM) | `GRU` |
| `-bi`, `--bi_directional_bit` | Whether to use bidirectional encoder | `True` |
| `-e_lay`, `--enc_layers` | Number of encoder layers | `2` |
| `-d_lay`, `--dec_layers` | Number of decoder layers | `2` |
| `-emd_size`, `--embedding_size` | Size of embeddings | `256` |
| `-h_size`, `--hidden_size` | Size of hidden states | `512` |
| `-attn`, `--attention_bit` | Whether to use attention mechanism | `True` |

#### Training Parameters

| Argument | Description | Default |
| --- | --- | --- |
| `-b_size`, `--batch_size` | Batch size for training | `64` |
| `-e_drp`, `--enc_dropout` | Encoder dropout probability | `0.2` |
| `-d_drp`, `--dec_dropout` | Decoder dropout probability | `0.2` |
| `-epoch`, `--max_epochs` | Maximum number of training epochs | `15` |
| `-lr`, `--learning_rate` | Learning rate for optimizer | `1e-3` |

#### Weights & Biases Parameters

| Argument | Description | Default |
| --- | --- | --- |
| `-wp`, `--wandb_project` | Weights & Biases project name | `DL_assignment_3` |
| `-use_wandb`, `--use_wandb` | Whether to log metrics to Weights & Biases | `False` |

#### Hyperparameter Sweep

| Argument | Description | Default |
| --- | --- | --- |
| `-sweep`, `--run_sweep` | Whether to run hyperparameter sweep | `False` |
| `-sweep_count`, `--sweep_count` | Number of runs for hyperparameter sweep | `20` |

#### Miscellaneous

| Argument | Description | Default |
| --- | --- | --- |
| `-seed`, `--seed` | Random seed for reproducibility | `42` |

### Examples

#### Train a model for Hindi transliteration with GRU cells:

```bash
python train.py -lang hin -ct GRU -bi True -attn True -epoch 20
```

#### Train a model with LSTM cells and no attention:

```bash
python train.py -ct LSTM -attn False -e_lay 3 -d_lay 3 -h_size 1024
```

#### Run hyperparameter sweep:

```bash
python train.py -sweep True -sweep_count 30 -use_wandb True
```

## Model Architecture

### Encoder-Decoder with Attention

![Model Architecture](https://img.shields.io/badge/Architecture-Encoder--Decoder%20with%20Attention-purple)

The model consists of:

1. **Encoder**: Converts the input sequence into a set of hidden representations
   - Optional bidirectional processing
   - Configurable number of layers
   - Choice of RNN, GRU, or LSTM cells

2. **Attention Mechanism**: Allows the decoder to focus on different parts of the input sequence
   - Bahdanau (additive) attention

3. **Decoder**: Generates the output sequence one token at a time
   - Configurable number of layers
   - Choice of RNN, GRU, or LSTM cells
   - Optional teacher forcing during training

## Results

After training completes, the script will output:
- Training and validation loss curves
- Accuracy on train, validation, and test sets
- Sample predictions
- Trained model weights

All outputs are saved to the specified output directory (default: `outputs/`).

## Hyperparameter Tuning

This project supports hyperparameter tuning using Weights & Biases. To run a hyperparameter sweep:

```bash
python train.py -sweep True -sweep_count 20 -use_wandb True
```

This will perform a Bayesian optimization search over the parameter space.



