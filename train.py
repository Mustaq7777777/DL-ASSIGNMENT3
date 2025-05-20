# train.py
# Main script for training sequence-to-sequence transliteration models

import torch
import argparse
import os
import sys

# Add current directory to path for importing local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from data_utils import download_and_extract_dataset, load_dakshina_data, prepare_matrices
from models import Encoder, Decoder, AttentionDecoder, Seq2Seq 
from train_utils import train_and_evaluate, accuracy_fun, vectors_to_actual_words, save_to_csv
from sweep_utils import seed_everything, init_sweep, create_sweep_objective, run_sweep_agent

# Get device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def argument_parsing():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Training Model for Transliteration')
    
    # Dataset parameters
    parser.add_argument('-lang', '--language', type=str, default='tel',
                        help='Language code (e.g., tel for Telugu)')
    parser.add_argument('-o', '--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--base_path', type=str, default=None,
                        help='Base path to the Dakshina dataset')
    
    # Model architecture parameters
    parser.add_argument('-ct', '--cell_type', type=str, default='GRU',
                        choices=['RNN', 'GRU', 'LSTM'],
                        help='RNN cell type')
    parser.add_argument('-bi', '--bi_directional_bit', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='Whether to use bidirectional encoder')
    parser.add_argument('-e_lay', '--enc_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('-d_lay', '--dec_layers', type=int, default=2,
                        help='Number of decoder layers')
    parser.add_argument('-emd_size', '--embedding_size', type=int, default=256,
                        help='Size of embeddings')
    parser.add_argument('-h_size', '--hidden_size', type=int, default=512,
                        help='Size of hidden states')
    parser.add_argument('-attn', '--attention_bit', type=lambda x: (str(x).lower() == 'true'), 
                        default=True, help='Whether to use attention mechanism')
    
    # Training parameters
    parser.add_argument('-b_size', '--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('-e_drp', '--enc_dropout', type=float, default=0.2,
                        help='Encoder dropout probability')
    parser.add_argument('-d_drp', '--dec_dropout', type=float, default=0.2,
                        help='Decoder dropout probability')
    parser.add_argument('-epoch', '--max_epochs', type=int, default=15,
                        help='Maximum number of training epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    
    # Weights & Biases parameters
    parser.add_argument('-wp', '--wandb_project', type=str, default='DL_assignment_3',
                        help='Weights & Biases project name')
    parser.add_argument('-use_wandb', '--use_wandb', type=lambda x: (str(x).lower() == 'true'), 
                        default=False, help='Whether to log metrics to Weights & Biases')
    
    # Hyperparameter sweep
    parser.add_argument('-sweep', '--run_sweep', type=lambda x: (str(x).lower() == 'true'), 
                        default=False, help='Whether to run hyperparameter sweep')
    parser.add_argument('-sweep_count', '--sweep_count', type=int, default=20,
                        help='Number of runs for hyperparameter sweep')
    
    # Misc
    parser.add_argument('-seed', '--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """
    Main function to run the training process.
    
    This is the entry point for the script.
    """
    # Parse arguments
    args = argument_parsing()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Download and extract dataset if needed
    dataset_dir = download_and_extract_dataset()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading {args.language} data...")
    data = load_dakshina_data(args.language, args.base_path)
    (eng_list_train, tel_list_train, eng_list_valid, tel_list_valid, 
     eng_list_test, tel_list_test, eng_vocab, tel_vocab, 
     max_eng_len, max_tel_len) = data
    
    # Prepare data matrices
    print("\nPreparing data matrices...")
    eng_matrix_train, tel_matrix_train = prepare_matrices(
        eng_list_train, tel_list_train, eng_vocab, tel_vocab, max_eng_len, max_tel_len
    )
    
    eng_matrix_valid, tel_matrix_valid = prepare_matrices(
        eng_list_valid, tel_list_valid, eng_vocab, tel_vocab, max_eng_len, max_tel_len
    )
    
    eng_matrix_test, tel_matrix_test = prepare_matrices(
        eng_list_test, tel_list_test, eng_vocab, tel_vocab, max_eng_len, max_tel_len
    )
    
    # Package data for ease of use
    data_tensors = (
        eng_matrix_train, tel_matrix_train,
        eng_matrix_valid, tel_matrix_valid,
        eng_matrix_test, tel_matrix_test
    )
    
    data_vocab = (eng_vocab, tel_vocab)
    
    # If running hyperparameter sweep
    if args.run_sweep:
        print("\nRunning hyperparameter sweep...")
        
        # Create objective function for sweep
        objective = create_sweep_objective(
            train_and_evaluate, data_tensors, data_vocab, args.language
        )
        
        # Initialize sweep
        sweep_id = init_sweep(
            with_attention=args.attention_bit,
            project=args.wandb_project
        )
        
        print(f"Sweep initialized with ID: {sweep_id}")
        print(f"Running {args.sweep_count} iterations...")
        
        # Run sweep
        run_sweep_agent(sweep_id, objective, args.sweep_count)
        
    else:
        # Train model with specified parameters
        print("\nTraining model with specified parameters...")
        
        model, accuracies, predictions = train_and_evaluate(
            args.cell_type,
            args.bi_directional_bit,
            args.embedding_size,
            args.enc_dropout,
            args.dec_dropout,
            args.enc_layers,
            args.dec_layers,
            args.hidden_size,
            args.batch_size,
            args.attention_bit,
            args.learning_rate,
            args.max_epochs,
            eng_matrix_train, tel_matrix_train,
            eng_matrix_valid, tel_matrix_valid,
            eng_matrix_test, tel_matrix_test,
            eng_vocab, tel_vocab,
            args.language,
            args.use_wandb,
            args.output_dir
        )
        
        # Save final model
        model_path = os.path.join(
            args.output_dir, 
            f"final_model_{args.cell_type}_{args.attention_bit}.pt"
        )
        torch.save(model.state_dict(), model_path)
        print(f"\nFinal model saved to: {model_path}")
        
        # Print final results
        print("\nFinal Results:")
        print(f"Train accuracy: {accuracies[0]:.2f}%")
        print(f"Valid accuracy: {accuracies[1]:.2f}%")
        print(f"Test accuracy: {accuracies[2]:.2f}%")
        
        # Print some example predictions
        print("\nExample predictions:")
        print(f"{'Source':<15} | {'Target':<15} | {'Predicted':<15}")
        print("-" * 48)
        for i in range(min(10, len(predictions))):
            src, pred, tgt = predictions[i]
            print(f"{src:<15} | {tgt:<15} | {pred:<15}")

if __name__ == "__main__":
    main()
