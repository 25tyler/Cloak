#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Foolbox Adversarial Attack Implementation
=========================================

This module provides a comprehensive implementation of adversarial attacks
using the Foolbox library. It includes various attack methods, visualization
tools, and utilities for creating adversarial examples on images.

Key Features:
- Multiple attack methods (FGSM, PGD, DeepFool, C&W, etc.)
- Image preprocessing and postprocessing
- Attack success rate analysis
- Perturbation visualization
- Model robustness testing
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import foolbox
from typing import Tuple, List, Dict, Optional, Union
import os
import json
from pathlib import Path


class AdversarialAttackEngine:
    """
    Main class for performing adversarial attacks on images using Foolbox.
    
    This class provides a high-level interface for:
    - Loading and preprocessing images
    - Creating adversarial examples using various attack methods
    - Analyzing attack success and perturbation characteristics
    - Visualizing results
    """
    
    def __init__(self, model_name: str = 'resnet50', device: str = 'auto'):
        """
        Initialize the adversarial attack engine.
        
        Args:
            model_name: Name of the pre-trained model to use ('resnet50', 'vgg16', etc.)
            device: Device to run on ('cpu', 'cuda', or 'auto')
        """
        self.device = self._get_device(device)
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.fmodel = self._create_foolbox_model()
        self.transform = self._get_transform()
        
        print(f"Initialized AdversarialAttackEngine with {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the best device to use."""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                return torch.device('cpu')
        return torch.device(device)
    
    def _load_model(self, model_name: str) -> nn.Module:
        """Load a pre-trained model."""
        model_map = {
            'resnet50': models.resnet50,
            'resnet18': models.resnet18,
            'vgg16': models.vgg16,
            'alexnet': models.alexnet,
            'densenet121': models.densenet121,
            'mobilenet_v2': models.mobilenet_v2
        }
        
        if model_name not in model_map:
            raise ValueError(f"Model {model_name} not supported. Available: {list(model_map.keys())}")
        
        model = model_map[model_name](pretrained=True)
        model.eval()
        model.to(self.device)
        return model
    
    def _create_foolbox_model(self) -> foolbox.PyTorchModel:
        """Create a Foolbox model wrapper."""
        return foolbox.PyTorchModel(self.model, bounds=(0, 1), device=self.device)
    
    def _get_transform(self) -> transforms.Compose:
        """Get the image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    
    def load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def load_image_from_pil(self, pil_image: Image.Image) -> torch.Tensor:
        """
        Load and preprocess an image from PIL Image object.
        
        Args:
            pil_image: PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        image = pil_image.convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def predict(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Get model prediction for an image.
        
        Args:
            image: Input image tensor
            
        Returns:
            Tuple of (predicted_class, confidence)
        """
        with torch.no_grad():
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            return predicted_class.item(), confidence.item()
    
    def create_adversarial_example(self, 
                                 image: torch.Tensor, 
                                 target_label: int,
                                 attack_method: str = 'FGSM',
                                 **attack_params) -> Tuple[torch.Tensor, Dict]:
        """
        Create an adversarial example using the specified attack method.
        
        Args:
            image: Input image tensor
            target_label: Target class for the attack
            attack_method: Attack method to use
            **attack_params: Additional parameters for the attack
            
        Returns:
            Tuple of (adversarial_image, attack_info)
        """
        # Get original prediction
        orig_pred, orig_conf = self.predict(image)
        
        # Create attack
        attack = self._create_attack(attack_method, **attack_params)
        
        # Generate adversarial example
        try:
            adversarial_image = attack(self.fmodel, image, label=target_label)
            
            # Get adversarial prediction
            adv_pred, adv_conf = self.predict(adversarial_image)
            
            # Calculate perturbation
            perturbation = adversarial_image - image
            perturbation_norm = torch.norm(perturbation).item()
            
            attack_info = {
                'attack_method': attack_method,
                'original_prediction': orig_pred,
                'original_confidence': orig_conf,
                'adversarial_prediction': adv_pred,
                'adversarial_confidence': adv_conf,
                'perturbation_norm': perturbation_norm,
                'success': adv_pred != orig_pred,
                'target_label': target_label
            }
            
            return adversarial_image, attack_info
            
        except Exception as e:
            print(f"Attack failed: {e}")
            return image, {'error': str(e)}
    
    def _create_attack(self, method: str, **params) -> foolbox.attacks.Attack:
        """Create a Foolbox attack object."""
        attack_map = {
            'FGSM': lambda: foolbox.attacks.FGSM(epsilons=params.get('epsilons', 0.1)),
            'PGD': lambda: foolbox.attacks.PGD(
                epsilons=params.get('epsilons', 0.1),
                steps=params.get('steps', 40),
                step_size=params.get('step_size', 0.01)
            ),
            'DeepFool': lambda: foolbox.attacks.DeepFoolAttack(),
            'C&W': lambda: foolbox.attacks.CarliniWagnerL2Attack(),
            'L2': lambda: foolbox.attacks.L2Attack(),
            'Linf': lambda: foolbox.attacks.LinfAttack(),
            'Boundary': lambda: foolbox.attacks.BoundaryAttack(),
            'HopSkipJump': lambda: foolbox.attacks.HopSkipJumpAttack()
        }
        
        if method not in attack_map:
            raise ValueError(f"Attack method {method} not supported. Available: {list(attack_map.keys())}")
        
        return attack_map[method]()
    
    def batch_attack(self, 
                    images: List[torch.Tensor], 
                    target_labels: List[int],
                    attack_method: str = 'FGSM',
                    **attack_params) -> List[Tuple[torch.Tensor, Dict]]:
        """
        Perform adversarial attacks on a batch of images.
        
        Args:
            images: List of input image tensors
            target_labels: List of target labels
            attack_method: Attack method to use
            **attack_params: Additional parameters for the attack
            
        Returns:
            List of (adversarial_image, attack_info) tuples
        """
        results = []
        for image, target_label in zip(images, target_labels):
            adv_image, info = self.create_adversarial_example(
                image, target_label, attack_method, **attack_params
            )
            results.append((adv_image, info))
        return results
    
    def analyze_robustness(self, 
                          images: List[torch.Tensor],
                          attack_methods: List[str] = None,
                          **attack_params) -> Dict:
        """
        Analyze model robustness against multiple attack methods.
        
        Args:
            images: List of input image tensors
            attack_methods: List of attack methods to test
            **attack_params: Additional parameters for attacks
            
        Returns:
            Dictionary with robustness analysis results
        """
        if attack_methods is None:
            attack_methods = ['FGSM', 'PGD', 'DeepFool']
        
        results = {}
        
        for method in attack_methods:
            print(f"Testing robustness against {method}...")
            method_results = []
            
            for i, image in enumerate(images):
                # Get original prediction
                orig_pred, _ = self.predict(image)
                
                # Create adversarial example
                adv_image, info = self.create_adversarial_example(
                    image, orig_pred, method, **attack_params
                )
                
                method_results.append(info)
            
            # Calculate statistics
            success_rate = sum(1 for r in method_results if r.get('success', False)) / len(method_results)
            avg_perturbation = np.mean([r.get('perturbation_norm', 0) for r in method_results])
            
            results[method] = {
                'success_rate': success_rate,
                'avg_perturbation': avg_perturbation,
                'individual_results': method_results
            }
        
        return results
    
    def visualize_attack(self, 
                        original: torch.Tensor,
                        adversarial: torch.Tensor,
                        attack_info: Dict,
                        save_path: Optional[str] = None) -> None:
        """
        Visualize the adversarial attack results.
        
        Args:
            original: Original image tensor
            adversarial: Adversarial image tensor
            attack_info: Attack information dictionary
            save_path: Optional path to save the visualization
        """
        # Convert tensors to numpy for visualization
        orig_np = original.squeeze().cpu().permute(1, 2, 0).numpy()
        adv_np = adversarial.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Calculate perturbation
        perturbation = adversarial - original
        pert_np = perturbation.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(orig_np)
        axes[0, 0].set_title(f'Original\nPred: {attack_info.get("original_prediction", "N/A")} '
                           f'(Conf: {attack_info.get("original_confidence", 0):.3f})')
        axes[0, 0].axis('off')
        
        # Adversarial image
        axes[0, 1].imshow(adv_np)
        axes[0, 1].set_title(f'Adversarial\nPred: {attack_info.get("adversarial_prediction", "N/A")} '
                           f'(Conf: {attack_info.get("adversarial_confidence", 0):.3f})')
        axes[0, 1].axis('off')
        
        # Perturbation (amplified for visibility)
        pert_vis = np.clip(pert_np * 10 + 0.5, 0, 1)
        axes[1, 0].imshow(pert_vis)
        axes[1, 0].set_title(f'Perturbation (×10)\nNorm: {attack_info.get("perturbation_norm", 0):.4f}')
        axes[1, 0].axis('off')
        
        # Difference
        diff = np.abs(adv_np - orig_np)
        axes[1, 1].imshow(diff)
        axes[1, 1].set_title(f'Absolute Difference\nMax: {diff.max():.4f}')
        axes[1, 1].axis('off')
        
        # Add overall title
        method = attack_info.get('attack_method', 'Unknown')
        success = "✓" if attack_info.get('success', False) else "✗"
        fig.suptitle(f'{method} Attack {success}', fontsize=16)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, filepath: str) -> None:
        """Save attack results to a JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filepath}")


def main():
    """Example usage of the AdversarialAttackEngine."""
    print("Foolbox Adversarial Attack Implementation")
    print("=" * 50)
    
    # Initialize the attack engine
    engine = AdversarialAttackEngine(model_name='resnet50')
    
    # Example: Load an image (you'll need to provide a real image path)
    # image_path = "path/to/your/image.jpg"
    # image = engine.load_image(image_path)
    
    # Example: Create adversarial example
    # target_label = 0  # Target class
    # adversarial_image, attack_info = engine.create_adversarial_example(
    #     image, target_label, attack_method='FGSM', epsilons=0.1
    # )
    
    # Example: Visualize results
    # engine.visualize_attack(image, adversarial_image, attack_info)
    
    print("AdversarialAttackEngine initialized successfully!")
    print("Use the methods to create adversarial examples on your images.")


if __name__ == "__main__":
    main()
