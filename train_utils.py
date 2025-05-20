# train_utils.py
# This file contains functions for training, evaluation, and visualization

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import wandb
import pandas as pd

# Get device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def accuracy_fun(eng_matrix, tel_matrix, batch_size, model):
    """
    Compute exact match accuracy on a dataset.
    
    This function computes the percentage of examples where the model's output
    exactly matches the target sequence.
    
    Args:
        eng_matrix (Tensor): Matrix of English word vectors
        tel_matrix (Tensor): Matrix of target language word vectors
        batch_size (int): Batch size for processing
        model (nn.Module): The trained model
        
    Returns:
        float: Accuracy percentage
    """
    correct = 0
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        # Process data in batches
        for batch_id in range((len(eng_matrix) + batch_size - 1) // batch_size):  # Ceiling division
            # Get batch
            start_idx = batch_size * batch_id
            end_idx = min(batch_size * (batch_id + 1), len(eng_matrix))
            inp_word = eng_matrix[start_idx:end_idx].to(device=device)
            out_word = tel_matrix[start_idx:end_idx].to(device=device)
            
            # Get actual batch size (may be smaller for last batch)
            actual_batch_size = inp_word.size(0)
            
            # Skip batch if it's empty
            if actual_batch_size == 0:
                continue
            
            # Transpose for sequence-first format
            inp_word = inp_word.T
            out_word = out_word.T
            
            # Forward pass with no teacher forcing
            output = model.forward(inp_word, out_word, 0)
            
            # Get predictions
            output = nn.Softmax(dim=2)(output)
            output = torch.argmax(output, dim=2)
            
            # Transpose back to batch-first for comparison
            output = output.T
            out_word = out_word.T
            
            # Count correct predictions (exact match of entire sequence)
            for i in range(actual_batch_size):
                if torch.equal(output[i][1:], out_word[i][1:]):
                    correct += 1
    
    # Return accuracy percentage
    return (correct * 100) / len(eng_matrix) if len(eng_matrix) > 0 else 0.0

def vectors_to_actual_words(model, eng_matrix, tel_matrix, batch_size, eng_vocab, tel_vocab, data_type):
    """
    Convert model predictions to readable words.
    
    This function generates predictions from the model and converts them
    back to readable text.
    
    Args:
        model (nn.Module): The trained model
        eng_matrix (Tensor): Matrix of English word vectors
        tel_matrix (Tensor): Matrix of target language word vectors
        batch_size (int): Batch size for processing
        eng_vocab (list): English vocabulary
        tel_vocab (list): Target language vocabulary
        data_type (str): Data split identifier (e.g., 'Train', 'Valid', 'Test')
        
    Returns:
        list: Tuples of (input word, predicted word, target word)
    """
    results = []
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        # Process data in batches
        for batch_id in range((len(eng_matrix) + batch_size - 1) // batch_size):  # Ceiling division
            # Get batch
            start_idx = batch_size * batch_id
            end_idx = min(batch_size * (batch_id + 1), len(eng_matrix))
            input_batch = eng_matrix[start_idx:end_idx].to(device=device)
            output_batch = tel_matrix[start_idx:end_idx].to(device=device)
            
            # Skip empty batches
            if input_batch.size(0) == 0:
                continue
            
            # Forward pass
            model_output = model.forward(input_batch.T, output_batch.T, 0)
            model_output = nn.Softmax(dim=2)(model_output)
            model_output = torch.argmax(model_output, dim=2)
            model_output = model_output.T
            
            # Process each example in the batch
            for idx in range(len(output_batch)):
                # Get the target and predicted sequences
                res_word = output_batch[idx]
                pred_word = model_output[idx]
                inp_word = input_batch[idx]
                
                # Convert to strings
                word_res = ""
                word_pred = ""
                word_inp = ""
                
                # Convert prediction to string
                for i in range(len(pred_word)):
                    if pred_word[i] > 0 and pred_word[i] < len(tel_vocab) + 1:
                        word_pred += tel_vocab[pred_word[i] - 1]
                
                # Convert input to string
                for i in range(len(inp_word)):
                    if inp_word[i] > 0 and inp_word[i] < len(eng_vocab) + 1:
                        word_inp += eng_vocab[inp_word[i] - 1]
                
                # Convert target to string
                for i in range(len(res_word)):
                    if res_word[i] > 0 and res_word[i] < len(tel_vocab) + 1:
                        word_res += tel_vocab[res_word[i] - 1]
                
                # Store the results
                results.append((word_inp, word_pred, word_res))
    
    return results

def save_to_csv(results, filename):
    """
    Save prediction results to a CSV file.
    
    Args:
        results (list): List of (input, prediction, target) tuples
        filename (str): Output file path
        
    Returns:
        str: Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    # Write results to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("Source,Predicted,Target\n")
        for src, pred, tgt in results:
            f.write(f"{src},{pred},{tgt}\n")
    
    print(f"Results saved to {filename}")
    return filename

def visualize_attention(attention_weights, input_tokens, output_tokens, filename=None):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights (Tensor): Attention weights matrix
        input_tokens (list): Input tokens
        output_tokens (list): Output tokens
        filename (str, optional): Path to save the visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = plt.axes()
    im = ax.imshow(attention_weights, cmap='viridis')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(input_tokens)))
    ax.set_yticks(np.arange(len(output_tokens)))
    ax.set_xticklabels(input_tokens)
    ax.set_yticklabels(output_tokens)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add title and labels
    plt.title('Attention Weights')
    plt.xlabel('Input')
    plt.ylabel('Output')
    
    # Save figure if a filename is provided
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    
    plt.show()

def compare_models_chart(model_results, title="Model Comparison", filename=None):
    """
    Create a bar chart comparing multiple models.
    
    Args:
        model_results (dict): Dictionary mapping model names to accuracy tuples (train, valid, test)
        title (str): Chart title
        filename (str, optional): Path to save the chart
    """
    labels = ['Train', 'Valid', 'Test']
    x = np.arange(len(labels))
    width = 0.8 / len(model_results)  # Width of bars adjusted for number of models
    
    plt.figure(figsize=(12, 8))
    
    # Plot bars for each model
    for i, (model_name, accuracies) in enumerate(model_results.items()):
        offset = (i - len(model_results)/2 + 0.5) * width
        bars = plt.bar(x + offset, accuracies, width, label=model_name)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8)
    
    plt.xlabel('Dataset Split')
    plt.ylabel('Accuracy (%)')
    plt.title(title)
    plt.xticks(x, labels)
    plt.legend(loc='best')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if filename:
        plt.savefig(filename)
    plt.show()

def train_and_evaluate(cell_type, bi_directional_bit, embedding_size, enc_dropout, 
                      dec_dropout, enc_layers, dec_layers, hidden_size, batch_size, 
                      attention_bit, learning_rate, max_epochs, eng_matrix_train, 
                      tel_matrix_train, eng_matrix_valid, tel_matrix_valid, 
                      eng_matrix_test, tel_matrix_test, eng_vocab, tel_vocab,
                      language='tel', use_wandb=False, output_dir='outputs'):
    """
    Train and evaluate a sequence-to-sequence model.
    
    This function handles the complete training and evaluation process,
    including logging metrics and saving results.
    
    Args:
        cell_type (str): RNN cell type ('RNN', 'GRU', or 'LSTM')
        bi_directional_bit (bool): Whether to use bidirectional encoder
        embedding_size (int): Size of embeddings
        enc_dropout (float): Encoder dropout probability
        dec_dropout (float): Decoder dropout probability
        enc_layers (int): Number of encoder layers
        dec_layers (int): Number of decoder layers
        hidden_size (int): Size of hidden states
        batch_size (int): Batch size for training
        attention_bit (bool): Whether to use attention mechanism
        learning_rate (float): Learning rate for optimizer
        max_epochs (int): Maximum number of training epochs
        eng_matrix_train (Tensor): Training data input matrix
        tel_matrix_train (Tensor): Training data target matrix
        eng_matrix_valid (Tensor): Validation data input matrix
        tel_matrix_valid (Tensor): Validation data target matrix
        eng_matrix_test (Tensor): Test data input matrix
        tel_matrix_test (Tensor): Test data target matrix
        eng_vocab (list): English vocabulary
        tel_vocab (list): Target language vocabulary
        language (str): Language code
        use_wandb (bool): Whether to log metrics to Weights & Biases
        output_dir (str): Directory to save outputs
        
    Returns:
        tuple: Trained model, accuracy metrics, and test predictions
    """
    # Import models module (to avoid circular import)
    from models import Encoder, Decoder, AttentionDecoder, Seq2Seq
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize wandb if requested
    if use_wandb:
        run_name = f"{cell_type}_{enc_layers}l_{embedding_size}e_{hidden_size}h_" \
                  f"{'attn' if attention_bit else 'no_attn'}_" \
                  f"{'bid' if bi_directional_bit else 'uni'}"
        
        wandb.init(
            project="DL_assignment_3",
            name=run_name,
            config={
                "cell_type": cell_type,
                "bi_directional": bi_directional_bit,
                "embedding_size": embedding_size,
                "enc_dropout": enc_dropout,
                "dec_dropout": dec_dropout,
                "enc_layers": enc_layers,
                "dec_layers": dec_layers,
                "hidden_size": hidden_size,
                "batch_size": batch_size,
                "attention": attention_bit,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "language": language
            }
        )
    
    # Model dimensions
    enc_input_size = len(eng_vocab) + 2  # +2 for special tokens
    dec_input_size = len(tel_vocab) + 2
    output_size = len(tel_vocab) + 2
    
    # Create encoder
    encoder_section = Encoder(
        enc_input_size, embedding_size, enc_layers, hidden_size,
        cell_type, bi_directional_bit, enc_dropout, batch_size
    ).to(device=device)
    
    # Create decoder (with or without attention)
    if attention_bit:
        decoder_section = AttentionDecoder(
            dec_input_size, embedding_size, hidden_size, output_size,
            cell_type, dec_layers, dec_dropout, bi_directional_bit
        ).to(device=device)
    else:
        decoder_section = Decoder(
            dec_input_size, embedding_size, hidden_size, dec_layers,
            dec_dropout, cell_type, output_size
        ).to(device=device)
    
    # Create sequence-to-sequence model
    model = Seq2Seq(
        decoder_section, encoder_section, cell_type, 
        bi_directional_bit, enc_layers, dec_layers
    ).to(device=device)
    
    # Print model summary
    print(f"Model: {cell_type} {'with' if attention_bit else 'without'} attention, "
          f"{'bidirectional' if bi_directional_bit else 'unidirectional'}")
    print(f"Encoder layers: {enc_layers}, Decoder layers: {dec_layers}")
    print(f"Embedding size: {embedding_size}, Hidden size: {hidden_size}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create loss function (ignoring padding)
    pad = len(tel_vocab) + 1
    loss_criterion = nn.CrossEntropyLoss(ignore_index=pad)
    
    # Store best validation accuracy
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 
                                   f"best_model_{cell_type}_{attention_bit}_{bi_directional_bit}.pt")
    
    # Training metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Main training loop
    print(f"\nStarting training for {max_epochs} epochs")
    
    for epoch in range(max_epochs):
        print(f"\nEpoch: {epoch+1}/{max_epochs}")
        
        # Set to training mode
        model.train()
        total_loss = 0
        step = 0
        
        # Training batches with progress bar
        batch_count = (len(eng_matrix_train) + batch_size - 1) // batch_size
        progress_bar = tqdm(range(batch_count), desc=f"Training {epoch+1}")
        
        for batch_id in progress_bar:
            # Get batch
            start_idx = batch_size * batch_id
            end_idx = min(batch_size * (batch_id + 1), len(eng_matrix_train))
            inp_word = eng_matrix_train[start_idx:end_idx].to(device=device)
            out_word = tel_matrix_train[start_idx:end_idx].to(device=device)
            
            # Skip empty batches
            if inp_word.size(0) == 0:
                continue
            
            # Transpose for sequence-first format
            out_word = out_word.T
            inp_word = inp_word.T
            
            # Forward pass
            output = model(inp_word, out_word)
            
            # Calculate loss (skip first token which is SOS)
            output = output[1:].reshape(-1, output.shape[2])
            out_word = out_word[1:].reshape(-1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = loss_criterion(output, out_word)
            total_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            
            # Update weights
            optimizer.step()
            
            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())
            step += 1
        
        # Calculate epoch average loss
        avg_loss = total_loss / step if step > 0 else float('inf')
        train_losses.append(avg_loss)
        print(f"Training loss: {avg_loss:.4f}")
        
        # Validate
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_id in range((len(eng_matrix_valid) + batch_size - 1) // batch_size):
                # Get batch
                start_idx = batch_size * batch_id
                end_idx = min(batch_size * (batch_id + 1), len(eng_matrix_valid))
                inp_word = eng_matrix_valid[start_idx:end_idx].to(device=device)
                out_word = tel_matrix_valid[start_idx:end_idx].to(device=device)
                
                # Skip empty batches
                if inp_word.size(0) == 0:
                    continue
                
                # Transpose for sequence-first format
                out_word = out_word.T
                inp_word = inp_word.T
                
                # Forward pass (no teacher forcing)
                output = model(inp_word, out_word, 0)
                
                # Calculate loss
                output = output[1:].reshape(-1, output.shape[2])
                out_word = out_word[1:].reshape(-1)
                loss = loss_criterion(output, out_word)
                val_loss += loss.item()
                val_steps += 1
        
        # Calculate validation loss
        avg_val_loss = val_loss / val_steps if val_steps > 0 else float('inf')
        val_losses.append(avg_val_loss)
        print(f"Validation loss: {avg_val_loss:.4f}")
        
        # Evaluate on train, validation, and test sets
        train_acc = accuracy_fun(eng_matrix_train, tel_matrix_train, batch_size, model)
        valid_acc = accuracy_fun(eng_matrix_valid, tel_matrix_valid, batch_size, model)
        test_acc = accuracy_fun(eng_matrix_test, tel_matrix_test, batch_size, model)
        
        train_accs.append(train_acc)
        val_accs.append(valid_acc)
        
        print(f"Train accuracy: {train_acc:.2f}%")
        print(f"Valid accuracy: {valid_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        
        # Save best model
        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved new best model with validation accuracy: {valid_acc:.2f}%")
        
        # Log to wandb if enabled
        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_acc,
                'valid_accuracy': valid_acc,
                'test_accuracy': test_acc
            })
    
    # Load best model for final evaluation
    if os.path.exists(best_model_path):
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path))
    
    # Final evaluation
    train_acc = accuracy_fun(eng_matrix_train, tel_matrix_train, batch_size, model)
    valid_acc = accuracy_fun(eng_matrix_valid, tel_matrix_valid, batch_size, model)
    test_acc = accuracy_fun(eng_matrix_test, tel_matrix_test, batch_size, model)
    
    print("\nFinal results:")
    print(f"Train accuracy: {train_acc:.2f}%")
    print(f"Valid accuracy: {valid_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Generate and save test predictions
    test_results = vectors_to_actual_words(
        model, eng_matrix_test, tel_matrix_test, batch_size, 
        eng_vocab, tel_vocab, 'Test'
    )
    
    # Save results to CSV
    results_file = os.path.join(output_dir, f"predictions_{cell_type}_{attention_bit}.csv")
    save_to_csv(test_results, results_file)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(output_dir, f"training_curves_{cell_type}_{attention_bit}.png")
    plt.savefig(plot_file)
    
    # Close wandb run if used
    if use_wandb:
        wandb.finish()
    
    return model, (train_acc, valid_acc, test_acc), test_results
