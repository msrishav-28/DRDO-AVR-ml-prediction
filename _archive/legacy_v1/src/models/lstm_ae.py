import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for unsupervised anomaly detection.
    Trains only on baseline data to learn 'normal' behavioral sequences.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=2, dropout=0.2):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.encoder_fc = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=input_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # --- Encoder ---
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        # We take the final hidden state of the LAST layer
        # h_n shape: (num_layers, batch, hidden_dim) -> h_n[-1]
        last_hidden = h_n[-1]
        
        # Compress to latent space
        latent = self.encoder_fc(last_hidden) 
        # latent shape: (batch_size, latent_dim)
        
        # --- Decoder ---
        # Expand latent back to hidden_dim
        dec_init = self.decoder_fc(latent)
        # dec_init shape: (batch_size, hidden_dim)
        
        # We need to provide the sequence length back to the decoder.
        # Repeat the initial decoded state across the sequence length.
        seq_len = x.size(1)
        dec_input = dec_init.unsqueeze(1).repeat(1, seq_len, 1)
        # dec_input shape: (batch_size, seq_len, hidden_dim)
        
        # Decode sequence
        dec_out, _ = self.decoder_lstm(dec_input)
        # dec_out shape: (batch_size, seq_len, input_dim)
        
        return dec_out

def calculate_reconstruction_error(model, x, criterion=nn.MSELoss(reduction='none')):
    """
    Calculates the sequence-wise reconstruction error to use as an anomaly score.
    """
    model.eval()
    with torch.no_grad():
        reconstructed = model(x)
        # Compute MSE per sequence per feature, then average across features and sequence length
        loss = criterion(reconstructed, x)
        # loss shape: (batch, seq_len, features). Mean over seq and features -> (batch,)
        anomaly_scores = loss.mean(dim=(1, 2))
    return anomaly_scores
