"""
Dataset class for evolution prediction.
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from ..models.lstm import LSTMConfig  # Import the config

class EvolutionDataset(Dataset):
    """Dataset class for cell evolution prediction."""
    
    def __init__(self, data_folder: str, predict_case: str, config: LSTMConfig):
        """
        Initialize dataset.
        
        Args:
            data_folder: Path to data directory
            predict_case: Type of prediction case
            config: LSTM model configuration
        """
        super().__init__()
        self.config = config
        
        # Load all property files
        property_files = {
            'topheight': f'all_topheight_{predict_case}.json',
            'altitude': f'all_altitude_{predict_case}.json',
            'area': f'all_area_{predict_case}.json',
            'smnr': f'all_smnr_{predict_case}.json',
            'smjr': f'all_smjr_{predict_case}.json',
            'mean2d': f'all_mean2d_{predict_case}.json',
            'path': f'all_path_{predict_case}.json'
        }
        
        # Load data from files
        self.properties = {}
        for prop_name, filename in property_files.items():
            with open(os.path.join(data_folder, filename), "rb") as f:
                self.properties[prop_name] = list(json.load(f).values())
        
        # Select target property
        property_dict = {
            "mean2d": self.properties['mean2d'],
            "smjr": self.properties['smjr'],
            "smnr": self.properties['smnr'],
            "area": self.properties['area']
        }
        self.y = property_dict[target_property]
        
        # Process data into sequences
        self._process_sequences()
    
    def _process_sequences(self):
        """Process raw data into input sequences and targets."""
        self.data = []
        self.target = []
        self.cellids = []
        
        # Process each cell's data into sequences
        for props in zip(self.properties['path'], self.properties['altitude'], 
                        self.properties['topheight'], self.properties['area'],
                        self.properties['smnr'], self.properties['smjr'],
                        self.properties['mean2d'], self.y):
            cell, alt, top, ar, nr, jr, me, y = props
            
            if len(cell) < 7:
                continue
                
            for t in range(3, len(cell)-3):
                self.data.append([cell[t-3], cell[t-2], cell[t-1], cell[t]])
                self.cellids.append([cell[t-3], cell[t-2], cell[t-1], cell[t], 
                                   cell[t+1], cell[t+2], cell[t+3]])
                self.target.append([y[t], y[t+1], y[t+2], y[t+3]])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, List]:
        """Get a single sample from the dataset."""
        cellid = self.data[idx]
        targetid = self.target[idx]
        x = []
        
        # Build input features based on weight_type
        for i in range(len(cellid)):
            features = self._get_features(idx, i)
            x.append(features)
            
        # Convert to tensors
        x = torch.tensor(np.array(x, dtype=np.float64))
        y = torch.tensor(np.array(targetid, dtype=np.float64)).unsqueeze(1)
        
        return x, y, self.cellids[idx]
    
    def _get_features(self, idx: int, i: int) -> List[float]:
        """Get features for a specific index and timestep based on weight_type."""
        area = self.properties['area'][idx][i]
        smjr = self.properties['smjr'][idx][i]
        smnr = self.properties['smnr'][idx][i]
        mean_reflect = self.properties['mean2d'][idx][i]
        centroidz85_fit = self.properties['altitude'][idx][i]
        echo_top_heigh = self.properties['topheight'][idx][i]
        
        if self.config.weight_type == "include-altitude":
            return [area, smjr, smnr, mean_reflect, centroidz85_fit, echo_top_heigh]
        elif self.config.weight_type == "only-altitude":
            target_value = {
                "mean2d": mean_reflect,
                "smjr": smjr,
                "smnr": smnr,
            }.get(self.config.target_property)
            return [centroidz85_fit, target_value]
        elif self.config.weight_type == "no-altitude":
            return [area, smjr, smnr, mean_reflect]
        else:
            raise ValueError(f"Invalid weight_type: {self.config.weight_type}")