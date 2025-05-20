# models.py
# This file contains neural network model definitions for the sequence-to-sequence transliteration task

import torch
import torch.nn as nn
import torch.nn.functional as func
import random

# Get device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder for sequence-to-sequence model.
    
    This class encodes input sequences into hidden representations that are 
    used by the decoder. It supports RNN, GRU, and LSTM cell types with
    optional bidirectional encoding.
    """
    def __init__(self, input_size, embedding_size, enc_layers, hidden_size, 
                 cell_type, bi_directional_bit, dropout, batch_size):
        """
        Initialize the encoder.
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_size (int): Size of the embeddings
            enc_layers (int): Number of encoder layers
            hidden_size (int): Size of the hidden state
            cell_type (str): Type of RNN cell ('RNN', 'GRU', or 'LSTM')
            bi_directional_bit (bool): Whether to use bidirectional encoder
            dropout (float): Dropout probability
            batch_size (int): Batch size
        """
        super(Encoder, self).__init__()
        
        # Store parameters as instance variables
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.enc_layers = enc_layers
        self.cell_type = cell_type
        self.bi_directional_bit = bi_directional_bit
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        # Initialize RNN based on cell type
        if cell_type == "RNN":    
            self.rnn = nn.RNN(embedding_size, hidden_size, enc_layers, 
                             dropout=dropout, bidirectional=bi_directional_bit)
        elif cell_type == "GRU":
            self.gru = nn.GRU(embedding_size, hidden_size, enc_layers, 
                             dropout=dropout, bidirectional=bi_directional_bit)
        else:  # LSTM
            self.lstm = nn.LSTM(embedding_size, hidden_size, enc_layers, 
                               dropout=dropout, bidirectional=bi_directional_bit)
    
    def forward(self, x, hidden, cell):
        """
        Forward pass through the encoder.
        
        Args:
            x (Tensor): Input sequence [seq_len, batch_size]
            hidden (Tensor): Initial hidden state
            cell (Tensor): Initial cell state (for LSTM)
            
        Returns:
            tuple: Output sequence and final hidden state (and cell state for LSTM)
        """
        # Get actual batch size from input
        actual_batch_size = x.shape[1]
        
        # Apply embedding and dropout
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        
        # Pass through the appropriate RNN type
        if self.cell_type == "RNN":
            output, hidden = self.rnn(embedding, hidden)
        elif self.cell_type == "GRU":
            output, hidden = self.gru(embedding, hidden)
        else:  # LSTM
            output, (hidden, cell) = self.lstm(embedding, (hidden, cell))
            return output, hidden, cell
        
        return output, hidden
    
    def initialize_hidden(self, batch_size=None):
        """
        Initialize hidden state tensor.
        
        This method creates a tensor of zeros for the initial hidden state.
        For bidirectional encoders, the tensor size is doubled.
        
        Args:
            batch_size (int, optional): Batch size to use. Defaults to self.batch_size.
            
        Returns:
            Tensor: Initial hidden state
        """
        # Use provided batch size if available, otherwise use default
        actual_batch_size = batch_size if batch_size is not None else self.batch_size
        
        # For bidirectional encoders, we need 2 * num_layers hidden states
        multiplier = 2 if self.bi_directional_bit else 1
        return torch.zeros(multiplier * self.enc_layers, actual_batch_size, 
                           self.hidden_size, device=device)
    
    def initialize_cell(self, batch_size=None):
        """
        Initialize cell state tensor (for LSTM).
        
        This method creates a tensor of zeros for the initial cell state.
        For bidirectional encoders, the tensor size is doubled.
        
        Args:
            batch_size (int, optional): Batch size to use. Defaults to self.batch_size.
            
        Returns:
            Tensor: Initial cell state
        """
        # Use provided batch size if available, otherwise use default
        actual_batch_size = batch_size if batch_size is not None else self.batch_size
        
        # For bidirectional encoders, we need 2 * num_layers cell states
        multiplier = 2 if self.bi_directional_bit else 1
        return torch.zeros(multiplier * self.enc_layers, actual_batch_size, 
                           self.hidden_size, device=device)

class Decoder(nn.Module):
    """
    Basic decoder without attention mechanism.
    
    This class decodes the encoder's hidden representations into
    output sequences one token at a time.
    """
    def __init__(self, input_size, embedding_size, hidden_size, dec_layers, 
                 dropout, cell_type, output_size):
        """
        Initialize the decoder.
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_size (int): Size of the embeddings
            hidden_size (int): Size of the hidden state
            dec_layers (int): Number of decoder layers
            dropout (float): Dropout probability
            cell_type (str): Type of RNN cell ('RNN', 'GRU', or 'LSTM')
            output_size (int): Size of the output vocabulary
        """
        super(Decoder, self).__init__()
        
        # Store parameters as instance variables
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.dec_layers = dec_layers
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.output_size = output_size
        
        # Initialize RNN based on cell type
        if cell_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, dec_layers, dropout=dropout)
        elif cell_type == "GRU":
            self.gru = nn.GRU(embedding_size, hidden_size, dec_layers, dropout=dropout)
        else:  # LSTM
            self.lstm = nn.LSTM(embedding_size, hidden_size, dec_layers, dropout=dropout)
        
        # Output projection layer
        self.fully_conc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, prev_output, prev_hidden, cell=None):
        """
        Forward pass through the decoder.
        
        Args:
            x (Tensor): Input token [batch_size]
            prev_output (Tensor): Previous encoder output
            prev_hidden (Tensor): Previous hidden state
            cell (Tensor, optional): Previous cell state (for LSTM)
            
        Returns:
            tuple: Output predictions and new hidden state (and cell state for LSTM)
        """
        # Reshape input token and apply embedding
        x = x.unsqueeze(0).int()  # Add sequence dimension: [1, batch_size]
        embedding = self.embedding(x)
        embedding = self.dropout(embedding)
        
        # Pass through the appropriate RNN type
        if self.cell_type == "RNN":
            outputs, hidden = self.rnn(embedding, prev_hidden)
        elif self.cell_type == "GRU":
            outputs, hidden = self.gru(embedding, prev_hidden)
        else:  # LSTM
            if cell is None:
                cell = torch.zeros_like(prev_hidden)
            outputs, (hidden, cell) = self.lstm(embedding, (prev_hidden, cell))
        
        # Project to vocabulary size
        pred = self.fully_conc(outputs)
        pred = pred.squeeze(0)  # Remove sequence dimension: [batch_size, output_size]
        
        # Return based on cell type
        if self.cell_type == "GRU" or self.cell_type == "RNN":
            return pred, hidden
        
        return pred, hidden, cell

class BahdanauAttention(nn.Module):
    """
    Bahdanau attention mechanism.
    
    This implements the attention mechanism from Bahdanau et al. (2015),
    "Neural Machine Translation by Jointly Learning to Align and Translate".
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        """
        Initialize the attention mechanism.
        
        Args:
            enc_hid_dim (int): Encoder hidden dimension
            dec_hid_dim (int): Decoder hidden dimension
        """
        super().__init__()
        
        # Attention layer for computing attention scores
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        
        # Vector for computing final attention weights
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        Compute attention weights.
        
        Args:
            hidden (Tensor): Decoder hidden state [batch_size, dec_hid_dim]
            encoder_outputs (Tensor): Encoder outputs [src_len, batch_size, enc_hid_dim]
            
        Returns:
            Tensor: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        # Repeat hidden state for each encoder position
        # [batch_size, dec_hid_dim] -> [batch_size, src_len, dec_hid_dim]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # Transpose encoder outputs for attention calculation
        # [src_len, batch_size, enc_hid_dim] -> [batch_size, src_len, enc_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # Calculate energy/attention scores
        # [batch_size, src_len, enc_hid_dim + dec_hid_dim] -> [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Convert to scalar scores
        # [batch_size, src_len, dec_hid_dim] -> [batch_size, src_len, 1] -> [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        
        # Apply softmax to get probability distribution
        # [batch_size, src_len]
        return func.softmax(attention, dim=1)

class AttentionDecoder(nn.Module):
    """
    Decoder with Bahdanau attention mechanism.
    
    This decoder attends to different parts of the encoder outputs
    when generating each output token.
    """
    def __init__(self, input_size, embedding_size, hidden_size, output_size, 
                 cell_type, dec_layers, dropout, bi_directional_bit):
        """
        Initialize the attention decoder.
        
        Args:
            input_size (int): Size of the input vocabulary
            embedding_size (int): Size of the embeddings
            hidden_size (int): Size of the hidden state
            output_size (int): Size of the output vocabulary
            cell_type (str): Type of RNN cell ('RNN', 'GRU', or 'LSTM')
            dec_layers (int): Number of decoder layers
            dropout (float): Dropout probability
            bi_directional_bit (bool): Whether encoder is bidirectional
        """
        super(AttentionDecoder, self).__init__()
        
        # Store parameters as instance variables
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell_type = cell_type
        self.dec_layers = dec_layers
        self.bi_directional_bit = bi_directional_bit
        self.embedding_size = embedding_size
        self.dropout = nn.Dropout(dropout)
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, embedding_size)
        
        # Attention mechanism - account for bidirectional encoder
        # If bidirectional, encoder outputs have 2x the hidden_size
        enc_hid_dim = hidden_size * 2 if bi_directional_bit else hidden_size
        self.attention = BahdanauAttention(enc_hid_dim, hidden_size)
        
        # RNN input dimension (embedding + context)
        self.rnn_input_dim = embedding_size + enc_hid_dim
        
        # Initialize RNN based on cell type
        if cell_type == "LSTM":
            self.lstm = nn.LSTM(self.rnn_input_dim, hidden_size, dec_layers, dropout=dropout)
        elif cell_type == "GRU":
            self.gru = nn.GRU(self.rnn_input_dim, hidden_size, dec_layers, dropout=dropout)
        else:  # RNN
            self.rnn = nn.RNN(self.rnn_input_dim, hidden_size, dec_layers, dropout=dropout)
        
        # Output projection (combines hidden state, context vector, and embedding)
        # Context vector size depends on bidirectional encoder
        self.fully_conc = nn.Linear(hidden_size + enc_hid_dim + embedding_size, output_size)
    
    def forward(self, x, encoder_outputs, prev_hidden, cell=None):
        """
        Forward pass with attention mechanism.
        
        Args:
            x (Tensor): Input token [batch_size]
            encoder_outputs (Tensor): Encoder outputs [src_len, batch_size, enc_hid_dim]
            prev_hidden (Tensor): Previous hidden state
            cell (Tensor, optional): Previous cell state (for LSTM)
            
        Returns:
            tuple: Output predictions and new hidden state (and cell state for LSTM)
        """
        # Get the last layer's hidden state for attention
        if self.cell_type == 'LSTM':
            # For LSTM, get the hidden state (not cell state)
            attention_hidden = prev_hidden[-1]
        else:
            # For GRU and RNN, just get the last layer
            attention_hidden = prev_hidden[-1]
        
        # Calculate attention weights
        attn_weights = self.attention(attention_hidden, encoder_outputs)
        
        # Create context vector by applying attention weights to encoder outputs
        # [batch_size, src_len] -> [batch_size, 1, src_len]
        attn_weights = attn_weights.unsqueeze(1)
        
        # [src_len, batch_size, enc_hid_dim] -> [batch_size, src_len, enc_hid_dim]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        # [batch_size, 1, src_len] x [batch_size, src_len, enc_hid_dim] -> [batch_size, 1, enc_hid_dim]
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # Embed input token
        x = x.unsqueeze(0)  # Add sequence dimension
        embedded = self.embedding(x)
        
        # Combine embedding and context for RNN input
        # [1, batch_size, emb_dim], [batch_size, 1, enc_hid_dim] -> [1, batch_size, emb_dim + enc_hid_dim]
        rnn_input = torch.cat((embedded, context.permute(1, 0, 2)), dim=2)
        
        # Pass through the appropriate RNN type
        if self.cell_type == "RNN":
            outputs, hidden = self.rnn(rnn_input, prev_hidden)
        elif self.cell_type == "GRU":
            outputs, hidden = self.gru(rnn_input, prev_hidden)
        else:  # LSTM
            if cell is None:
                cell = torch.zeros_like(prev_hidden)
            outputs, (hidden, cell) = self.lstm(rnn_input, (prev_hidden, cell))
        
        # For output projection, combine hidden state, context, and embedded input
        outputs = outputs.squeeze(0)  # Remove sequence dimension
        embedded = embedded.squeeze(0)  # Remove sequence dimension
        context = context.squeeze(1)   # Remove extra dimension
        
        # Project to vocabulary size
        pred = self.fully_conc(torch.cat((outputs, context, embedded), dim=1))
        
        # Return based on cell type
        if self.cell_type == "GRU" or self.cell_type == "RNN":
            return pred, hidden
        else:
            return pred, hidden, cell

class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model combining encoder and decoder.
    
    This model encodes an input sequence and then decodes it into an output sequence.
    It supports teacher forcing during training.
    """
    def __init__(self, decoder, encoder, cell_type, bidirectional_bit, 
                 encoder_layers, decoder_layers):
        """
        Initialize the sequence-to-sequence model.
        
        Args:
            decoder (nn.Module): Decoder module
            encoder (nn.Module): Encoder module
            cell_type (str): Type of RNN cell ('RNN', 'GRU', or 'LSTM')
            bidirectional_bit (bool): Whether encoder is bidirectional
            encoder_layers (int): Number of encoder layers
            decoder_layers (int): Number of decoder layers
        """
        super(Seq2Seq, self).__init__()
        
        # Store components and parameters
        self.decoder = decoder
        self.encoder = encoder
        self.cell_type = cell_type
        self.bidirectional_bit = bidirectional_bit
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
    
    def forward(self, input_seq, target, teacher_force_ratio=0.5):
        """
        Forward pass through the sequence-to-sequence model.
        
        Args:
            input_seq (Tensor): Input sequence [seq_len, batch_size]
            target (Tensor): Target sequence [seq_len, batch_size]
            teacher_force_ratio (float): Probability of using teacher forcing
            
        Returns:
            Tensor: Output predictions [seq_len, batch_size, output_size]
        """
        # Get sequence and batch dimensions
        batch_size = input_seq.shape[1]
        tar_seq_length = target.shape[0]
        final_target_vocab_size = self.decoder.output_size
        
        # Initialize outputs tensor
        outputs = torch.zeros(tar_seq_length, batch_size, 
                             final_target_vocab_size).to(device=device)
        
        # Initialize encoder states with actual batch size
        hidden = self.encoder.initialize_hidden(batch_size)
        cell = self.encoder.initialize_cell(batch_size)
        
        # Encode input sequence
        if self.cell_type == "RNN" or self.cell_type == "GRU":
            encoder_output, hidden = self.encoder(input_seq, hidden, cell)
        else:  # LSTM
            encoder_output, hidden, cell = self.encoder(input_seq, hidden, cell)
        
        # Handle bidirectional encoder or different layer counts
        if self.decoder_layers != self.encoder_layers or self.bidirectional_bit:
            if self.cell_type in ["RNN", "GRU", "LSTM"]:
                # Convert bidirectional encoder hidden states
                if self.bidirectional_bit:
                    # Combine forward and backward directions
                    hidden_forward = hidden[:self.encoder_layers]
                    hidden_backward = hidden[self.encoder_layers:]
                    # Sum the forward and backward hidden states
                    hidden_combined = hidden_forward + hidden_backward
                    
                    # Create a new tensor for the decoder
                    hidden_decoder = torch.zeros(self.decoder_layers, batch_size, 
                                              self.encoder.hidden_size, device=device)
                    
                    # Fill the decoder hidden state
                    for i in range(self.decoder_layers):
                        if i < self.encoder_layers:
                            hidden_decoder[i] = hidden_combined[i]
                        else:
                            # Repeat the last layer for extra decoder layers
                            hidden_decoder[i] = hidden_combined[-1]
                    
                    hidden = hidden_decoder
                
                # Match decoder layers if needed (without bidirectional)
                elif self.decoder_layers > self.encoder_layers:
                    hidden_decoder = torch.zeros(self.decoder_layers, batch_size, 
                                             self.encoder.hidden_size, device=device)
                    
                    for i in range(self.decoder_layers):
                        if i < self.encoder_layers:
                            hidden_decoder[i] = hidden[i]
                        else:
                            hidden_decoder[i] = hidden[-1]
                    
                    hidden = hidden_decoder
            
            # Handle cell states for LSTM
            if self.cell_type == "LSTM":
                if self.bidirectional_bit:
                    # Combine forward and backward directions
                    cell_forward = cell[:self.encoder_layers]
                    cell_backward = cell[self.encoder_layers:]
                    cell_combined = cell_forward + cell_backward
                    
                    # Create a new tensor for the decoder
                    cell_decoder = torch.zeros(self.decoder_layers, batch_size, 
                                           self.encoder.hidden_size, device=device)
                    
                    # Fill the decoder cell state
                    for i in range(self.decoder_layers):
                        if i < self.encoder_layers:
                            cell_decoder[i] = cell_combined[i]
                        else:
                            cell_decoder[i] = cell_combined[-1]
                    
                    cell = cell_decoder
                
                # Match decoder layers if needed (without bidirectional)
                elif self.decoder_layers > self.encoder_layers:
                    cell_decoder = torch.zeros(self.decoder_layers, batch_size, 
                                          self.encoder.hidden_size, device=device)
                    
                    for i in range(self.decoder_layers):
                        if i < self.encoder_layers:
                            cell_decoder[i] = cell[i]
                        else:
                            cell_decoder[i] = cell[-1]
                    
                    cell = cell_decoder
        
        # Start with first token (SOS token)
        x = target[0]
        
        # Generate sequence token by token
        for t in range(1, tar_seq_length):
            # Process through decoder
            if self.cell_type == "RNN" or self.cell_type == "GRU":
                output, hidden = self.decoder(x, encoder_output, hidden)
            else:  # LSTM
                output, hidden, cell = self.decoder(x, encoder_output, hidden, cell)
            
            # Store output
            outputs[t] = output
            
            # Teacher forcing: use target token with probability teacher_force_ratio
            if random.random() < teacher_force_ratio:
                x = target[t]  # Use ground truth token
            else:
                # Otherwise use model's prediction
                predicted = output.argmax(1)
                x = predicted
        
        return outputs
