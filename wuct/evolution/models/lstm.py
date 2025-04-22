"""
LSTM-based evolution prediction model.
"""

import os
import random
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class LSTMConfig:
    """Configuration for LSTM model and prediction."""
    weight_type: str
    target_property: str
    hidden_size: int = 32
    num_layers: int = 2
    dropout: float = 0.0
    seq_len: int = 3
    
    # Weight type configurations
    WEIGHT_TYPES = {
        "include-altitude": {"input_size": 6, "file_suffix": "_all"},
        "no-altitude": {"input_size": 4, "file_suffix": ""},
        "only-altitude": {"input_size": 2, "file_suffix": "_zand"}
    }
    
    @property
    def input_size(self) -> int:
        """Get input size based on weight type."""
        return self.WEIGHT_TYPES[self.weight_type]["input_size"]
    
    def get_weight_path(self, model_dir: str) -> str:
        """Get path to model weights file."""
        suffix = self.WEIGHT_TYPES[self.weight_type]["file_suffix"]
        if suffix == "_zand":
            suffix = f"_zand{self.target_property}"
            
        pth_name = self.target_property.upper() if self.target_property != "mean2d" else "MEAN2d"
        filename = f"LSTM_{pth_name}_regression{suffix}.pth"
        return os.path.join(model_dir, self.target_property, filename)

class LSTMModel(nn.Module):
    """LSTM-based sequence-to-sequence model for evolution prediction."""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=1,  # Single feature input
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )
        self.fc = nn.Linear(config.hidden_size, 1)
    
    def forward(self, x: torch.Tensor, target: Optional[torch.Tensor] = None,
                teacher_force_ratio: float = 0.0) -> torch.Tensor:
        batch_size = x.shape[0]
        outputs = torch.zeros(batch_size, self.config.seq_len, 1).to(x.device)
        
        # Encode
        _, (hidden, cell) = self.encoder_lstm(x)
        
        # Initial decoder input
        decoder_input = target[:, 0, :] if target is not None else x[:, -1, :1]
        
        # Decode
        for t in range(self.config.seq_len):
            output, (hidden, cell) = self._decode_step(decoder_input, hidden, cell)
            outputs[:, t, :] = output
            
            # Teacher forcing
            decoder_input = target[:, t+1, :] if (target is not None and 
                random.random() < teacher_force_ratio) else output
        
        return outputs
    
    def _decode_step(self, x: torch.Tensor, hidden: torch.Tensor, 
                     cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.unsqueeze(1)  # Add sequence dimension
        output, (hidden, cell) = self.decoder_lstm(x, (hidden, cell))
        prediction = self.fc(output.squeeze(1))
        return prediction, (hidden, cell)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make predictions without teacher forcing."""
        return self.forward(x, target=None, teacher_force_ratio=0.0)
    
    def save(self, path: str) -> None:
        """Save model state to file."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: str) -> None:
        """Load model state from file."""
        self.load_state_dict(torch.load(path))