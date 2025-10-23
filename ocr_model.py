#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Model for Adversarial Attacks
================================

This module provides a specialized OCR model for targeting adversarial attacks
on text glyphs. It's designed to be confused by adversarial perturbations
applied to individual characters.

The model is trained to recognize characters but can be easily fooled
by small perturbations, making it perfect for generating adversarial examples
that resist OCR and text recognition systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple
import numpy as np


class OCRModel(nn.Module):
    """
    OCR model specifically designed for adversarial attacks on text glyphs.
    
    This model is trained to recognize individual characters but is vulnerable
    to adversarial perturbations, making it ideal for generating adversarial
    examples that resist OCR systems.
    """
    
    def __init__(self, num_classes: int = 95, input_size: int = 64):
        """
        Initialize the OCR model.
        
        Args:
            num_classes: Number of character classes (default: 95 for ASCII printable)
            input_size: Input image size (default: 64x64)
        """
        super(OCRModel, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Use a lightweight CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def _calculate_flattened_size(self) -> int:
        """Calculate the size after convolution and pooling operations."""
        # Simulate forward pass to calculate size
        x = torch.zeros(1, 1, self.input_size, self.input_size)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the OCR model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Convolutional layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class CharacterClassifier:
    """
    Character classifier that uses the OCR model for adversarial attacks.
    
    This class provides a high-level interface for character recognition
    and adversarial attack generation.
    """
    
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize the character classifier.
        
        Args:
            model_path: Path to pre-trained model (if available)
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.model = OCRModel()
        self.model.to(self.device)
        
        # Character mapping for ASCII printable characters
        self.char_to_idx = self._create_char_mapping()
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # Load pre-trained weights if available
        if model_path and torch.load(model_path, map_location=self.device):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load pre-trained model: {e}")
                print("Using randomly initialized model")
        else:
            print("Using randomly initialized model")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _create_char_mapping(self) -> Dict[str, int]:
        """Create mapping from characters to indices."""
        # ASCII printable characters (32-126)
        char_mapping = {}
        for i in range(32, 127):
            char_mapping[chr(i)] = i - 32
        
        return char_mapping
    
    def predict_character(self, glyph_tensor: torch.Tensor) -> Tuple[str, float]:
        """
        Predict the character from a glyph tensor.
        
        Args:
            glyph_tensor: Input tensor of shape (1, 1, height, width)
            
        Returns:
            Tuple of (predicted_character, confidence)
        """
        self.model.eval()
        with torch.no_grad():
            logits = self.model(glyph_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_char = self.idx_to_char.get(predicted_idx.item(), '?')
            return predicted_char, confidence.item()
    
    def get_character_embeddings(self, glyph_tensor: torch.Tensor) -> torch.Tensor:
        """
        Get character embeddings for adversarial attack generation.
        
        Args:
            glyph_tensor: Input tensor of shape (1, 1, height, width)
            
        Returns:
            Embedding tensor
        """
        self.model.eval()
        with torch.no_grad():
            # Get features from the last fully connected layer
            x = glyph_tensor
            x = F.max_pool2d(F.relu(self.model.conv1(x)), 2)
            x = F.max_pool2d(F.relu(self.model.conv2(x)), 2)
            x = F.max_pool2d(F.relu(self.model.conv3(x)), 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.model.fc1(x))
            x = F.relu(self.model.fc2(x))
            return x
    
    def create_adversarial_glyph(self, 
                                glyph_tensor: torch.Tensor,
                                target_character: str = None,
                                epsilon: float = 0.1) -> Tuple[torch.Tensor, Dict]:
        """
        Create an adversarial glyph that fools the OCR model.
        
        Args:
            glyph_tensor: Original glyph tensor
            target_character: Target character to misclassify as
            epsilon: Perturbation strength
            
        Returns:
            Tuple of (adversarial_glyph, attack_info)
        """
        # Get original prediction
        orig_char, orig_conf = self.predict_character(glyph_tensor)
        
        # Set target
        if target_character is None:
            # Random target character
            target_idx = np.random.randint(0, len(self.idx_to_char))
            target_character = self.idx_to_char[target_idx]
        
        target_idx = self.char_to_idx.get(target_character, 0)
        
        # Create adversarial example using FGSM
        glyph_tensor.requires_grad = True
        
        # Forward pass
        logits = self.model(glyph_tensor)
        loss = F.cross_entropy(logits, torch.tensor([target_idx]).to(self.device))
        
        # Compute gradients
        loss.backward()
        
        # Create adversarial example
        adversarial_glyph = glyph_tensor + epsilon * glyph_tensor.grad.sign()
        adversarial_glyph = torch.clamp(adversarial_glyph, 0, 1)
        
        # Get adversarial prediction
        adv_char, adv_conf = self.predict_character(adversarial_glyph.detach())
        
        attack_info = {
            'original_character': orig_char,
            'original_confidence': orig_conf,
            'adversarial_character': adv_char,
            'adversarial_confidence': adv_conf,
            'target_character': target_character,
            'epsilon': epsilon,
            'success': adv_char != orig_char
        }
        
        return adversarial_glyph.detach(), attack_info


def create_synthetic_training_data(num_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create synthetic training data for the OCR model.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Generate random character
        char_code = np.random.randint(32, 127)
        char = chr(char_code)
        
        # Create simple character image
        img = np.random.rand(64, 64) * 0.1  # Background noise
        # Add character-like pattern (simplified)
        center_x, center_y = 32, 32
        for i in range(64):
            for j in range(64):
                if abs(i - center_x) < 10 and abs(j - center_y) < 15:
                    img[i, j] = 0.8 + np.random.rand() * 0.2
        
        images.append(img)
        labels.append(char_code - 32)  # Convert to 0-based index
    
    return torch.tensor(images).unsqueeze(1), torch.tensor(labels)


def train_ocr_model(model: OCRModel, 
                   num_epochs: int = 10,
                   learning_rate: float = 0.001) -> OCRModel:
    """
    Train the OCR model on synthetic data.
    
    Args:
        model: OCR model to train
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        Trained model
    """
    # Create synthetic training data
    images, labels = create_synthetic_training_data(1000)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Set up training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Creating OCR model for adversarial attacks...")
    
    # Create and train model
    model = OCRModel()
    trained_model = train_ocr_model(model)
    
    # Save model
    torch.save(trained_model.state_dict(), 'ocr_model.pth')
    print("OCR model trained and saved!")
    
    # Test character classification
    classifier = CharacterClassifier()
    
    # Create a test glyph (simplified)
    test_glyph = torch.rand(1, 1, 64, 64)
    char, conf = classifier.predict_character(test_glyph)
    print(f"Predicted character: {char} (confidence: {conf:.3f})")
    
    # Create adversarial example
    adv_glyph, attack_info = classifier.create_adversarial_glyph(test_glyph)
    print(f"Adversarial attack info: {attack_info}")
