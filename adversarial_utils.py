#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adversarial Attack Utilities
===========================

This module provides utility functions for image processing, visualization,
and analysis of adversarial attacks. These utilities complement the main
adversarial attack engine.

Key Features:
- Image preprocessing and augmentation
- Advanced visualization tools
- Attack analysis and metrics
- Model comparison utilities
- Batch processing helpers
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional, Union
import cv2
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def load_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> Image.Image:
        """Load and resize an image."""
        image = Image.open(image_path).convert('RGB')
        return image.resize(size, Image.Resampling.LANCZOS)
    
    @staticmethod
    def preprocess_for_model(image: Image.Image, 
                           normalize: bool = True) -> torch.Tensor:
        """Preprocess image for model input."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]) if normalize else transforms.Lambda(lambda x: x)
        ])
        return transform(image).unsqueeze(0)
    
    @staticmethod
    def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
        """Denormalize a tensor back to [0, 1] range."""
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return tensor * std + mean
    
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """Convert tensor to PIL Image."""
        # Clamp values to [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to numpy and transpose
        array = tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        # Convert to PIL
        return Image.fromarray((array * 255).astype(np.uint8))
    
    @staticmethod
    def apply_augmentation(image: Image.Image, 
                          augmentation_type: str = 'brightness') -> Image.Image:
        """Apply various image augmentations."""
        if augmentation_type == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1.2)
        elif augmentation_type == 'contrast':
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(1.2)
        elif augmentation_type == 'sharpness':
            enhancer = ImageEnhance.Sharpness(image)
            return enhancer.enhance(1.2)
        elif augmentation_type == 'blur':
            return image.filter(ImageFilter.GaussianBlur(radius=1))
        else:
            return image


class AttackAnalyzer:
    """Class for analyzing adversarial attack results."""
    
    def __init__(self):
        self.results = []
    
    def add_result(self, attack_info: Dict):
        """Add a single attack result for analysis."""
        self.results.append(attack_info)
    
    def add_batch_results(self, batch_results: List[Dict]):
        """Add multiple attack results for analysis."""
        self.results.extend(batch_results)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive attack metrics."""
        if not self.results:
            return {}
        
        # Basic metrics
        total_attacks = len(self.results)
        successful_attacks = sum(1 for r in self.results if r.get('success', False))
        success_rate = successful_attacks / total_attacks
        
        # Perturbation metrics
        perturbations = [r.get('perturbation_norm', 0) for r in self.results if 'perturbation_norm' in r]
        avg_perturbation = np.mean(perturbations) if perturbations else 0
        max_perturbation = np.max(perturbations) if perturbations else 0
        min_perturbation = np.min(perturbations) if perturbations else 0
        
        # Confidence metrics
        orig_confidences = [r.get('original_confidence', 0) for r in self.results]
        adv_confidences = [r.get('adversarial_confidence', 0) for r in self.results]
        
        return {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'success_rate': success_rate,
            'avg_perturbation': avg_perturbation,
            'max_perturbation': max_perturbation,
            'min_perturbation': min_perturbation,
            'avg_original_confidence': np.mean(orig_confidences),
            'avg_adversarial_confidence': np.mean(adv_confidences),
            'confidence_drop': np.mean(orig_confidences) - np.mean(adv_confidences)
        }
    
    def plot_attack_comparison(self, save_path: Optional[str] = None):
        """Create comparison plots for different attack methods."""
        if not self.results:
            print("No results to plot")
            return
        
        # Group results by attack method
        method_results = {}
        for result in self.results:
            method = result.get('attack_method', 'Unknown')
            if method not in method_results:
                method_results[method] = []
            method_results[method].append(result)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate comparison
        methods = list(method_results.keys())
        success_rates = []
        for method in methods:
            method_data = method_results[method]
            success_rate = sum(1 for r in method_data if r.get('success', False)) / len(method_data)
            success_rates.append(success_rate)
        
        axes[0, 0].bar(methods, success_rates)
        axes[0, 0].set_title('Attack Success Rate by Method')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Perturbation comparison
        perturbations_by_method = {}
        for method in methods:
            method_data = method_results[method]
            perturbations = [r.get('perturbation_norm', 0) for r in method_data if 'perturbation_norm' in r]
            perturbations_by_method[method] = perturbations
        
        axes[0, 1].boxplot(perturbations_by_method.values(), labels=perturbations_by_method.keys())
        axes[0, 1].set_title('Perturbation Magnitude by Method')
        axes[0, 1].set_ylabel('Perturbation Norm')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        all_orig_conf = [r.get('original_confidence', 0) for r in self.results]
        all_adv_conf = [r.get('adversarial_confidence', 0) for r in self.results]
        
        axes[1, 0].hist(all_orig_conf, alpha=0.7, label='Original', bins=20)
        axes[1, 0].hist(all_adv_conf, alpha=0.7, label='Adversarial', bins=20)
        axes[1, 0].set_title('Confidence Distribution')
        axes[1, 0].set_xlabel('Confidence')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Attack method frequency
        method_counts = {}
        for result in self.results:
            method = result.get('attack_method', 'Unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        axes[1, 1].pie(method_counts.values(), labels=method_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Attack Method Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive attack analysis report."""
        metrics = self.calculate_metrics()
        
        report = f"""
Adversarial Attack Analysis Report
==================================

Summary Statistics:
- Total Attacks: {metrics.get('total_attacks', 0)}
- Successful Attacks: {metrics.get('successful_attacks', 0)}
- Success Rate: {metrics.get('success_rate', 0):.2%}

Perturbation Analysis:
- Average Perturbation: {metrics.get('avg_perturbation', 0):.4f}
- Maximum Perturbation: {metrics.get('max_perturbation', 0):.4f}
- Minimum Perturbation: {metrics.get('min_perturbation', 0):.4f}

Confidence Analysis:
- Average Original Confidence: {metrics.get('avg_original_confidence', 0):.4f}
- Average Adversarial Confidence: {metrics.get('avg_adversarial_confidence', 0):.4f}
- Average Confidence Drop: {metrics.get('confidence_drop', 0):.4f}

Attack Method Breakdown:
"""
        
        # Add method-specific statistics
        method_stats = {}
        for result in self.results:
            method = result.get('attack_method', 'Unknown')
            if method not in method_stats:
                method_stats[method] = []
            method_stats[method].append(result)
        
        for method, results in method_stats.items():
            method_success = sum(1 for r in results if r.get('success', False)) / len(results)
            method_perturbation = np.mean([r.get('perturbation_norm', 0) for r in results if 'perturbation_norm' in r])
            
            report += f"""
{method}:
  - Success Rate: {method_success:.2%}
  - Average Perturbation: {method_perturbation:.4f}
  - Number of Attacks: {len(results)}
"""
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


class AdvancedVisualizer:
    """Advanced visualization tools for adversarial attacks."""
    
    @staticmethod
    def create_attack_grid(images: List[torch.Tensor], 
                          labels: List[str],
                          title: str = "Attack Comparison") -> None:
        """Create a grid visualization of multiple images."""
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, (image, label) in enumerate(zip(images, labels)):
            if i < len(axes):
                img_np = image.squeeze().cpu().permute(1, 2, 0).numpy()
                axes[i].imshow(img_np)
                axes[i].set_title(label)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(images), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def visualize_perturbation_heatmap(perturbation: torch.Tensor, 
                                     save_path: Optional[str] = None) -> None:
        """Create a heatmap visualization of the perturbation."""
        pert_np = perturbation.squeeze().cpu().numpy()
        
        # Calculate L2 norm across channels
        pert_magnitude = np.sqrt(np.sum(pert_np**2, axis=0))
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Individual channels
        for i in range(3):
            im = axes[i].imshow(pert_np[i], cmap='RdBu_r', vmin=-pert_np.max(), vmax=pert_np.max())
            axes[i].set_title(f'Channel {i+1}')
            axes[i].axis('off')
            plt.colorbar(im, ax=axes[i])
        
        # Add overall magnitude
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(pert_magnitude, cmap='hot')
        ax.set_title('Perturbation Magnitude (L2 Norm)')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        
        if save_path:
            fig.savefig(save_path.replace('.png', '_channels.png'), dpi=300, bbox_inches='tight')
            fig2.savefig(save_path.replace('.png', '_magnitude.png'), dpi=300, bbox_inches='tight')
            print(f"Perturbation heatmaps saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_attack_timeline(attack_results: List[Dict], 
                              save_path: Optional[str] = None) -> None:
        """Create a timeline visualization of attack progression."""
        if not attack_results:
            return
        
        # Extract metrics over time
        steps = list(range(len(attack_results)))
        success_rates = []
        perturbations = []
        confidences = []
        
        for result in attack_results:
            success_rates.append(1 if result.get('success', False) else 0)
            perturbations.append(result.get('perturbation_norm', 0))
            confidences.append(result.get('adversarial_confidence', 0))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate over time
        axes[0, 0].plot(steps, success_rates, 'o-')
        axes[0, 0].set_title('Attack Success Over Time')
        axes[0, 0].set_xlabel('Attack Step')
        axes[0, 0].set_ylabel('Success (1=Yes, 0=No)')
        axes[0, 0].grid(True)
        
        # Perturbation magnitude over time
        axes[0, 1].plot(steps, perturbations, 'o-', color='red')
        axes[0, 1].set_title('Perturbation Magnitude Over Time')
        axes[0, 1].set_xlabel('Attack Step')
        axes[0, 1].set_ylabel('Perturbation Norm')
        axes[0, 1].grid(True)
        
        # Confidence over time
        axes[1, 0].plot(steps, confidences, 'o-', color='green')
        axes[1, 0].set_title('Adversarial Confidence Over Time')
        axes[1, 0].set_xlabel('Attack Step')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(True)
        
        # Cumulative success rate
        cumulative_success = np.cumsum(success_rates) / np.arange(1, len(success_rates) + 1)
        axes[1, 1].plot(steps, cumulative_success, 'o-', color='purple')
        axes[1, 1].set_title('Cumulative Success Rate')
        axes[1, 1].set_xlabel('Attack Step')
        axes[1, 1].set_ylabel('Cumulative Success Rate')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Timeline visualization saved to {save_path}")
        
        plt.show()


def create_sample_dataset(num_samples: int = 10, 
                         image_size: Tuple[int, int] = (224, 224)) -> List[torch.Tensor]:
    """Create a sample dataset of random images for testing."""
    samples = []
    for _ in range(num_samples):
        # Create random image
        random_image = torch.randn(3, image_size[0], image_size[1])
        # Normalize to [0, 1] range
        random_image = (random_image - random_image.min()) / (random_image.max() - random_image.min())
        samples.append(random_image.unsqueeze(0))
    return samples


def benchmark_attacks(engine, 
                     images: List[torch.Tensor],
                     attack_methods: List[str] = None) -> Dict:
    """Benchmark different attack methods on a set of images."""
    if attack_methods is None:
        attack_methods = ['FGSM', 'PGD', 'DeepFool', 'C&W']
    
    benchmark_results = {}
    
    for method in attack_methods:
        print(f"Benchmarking {method}...")
        method_results = []
        
        for i, image in enumerate(images):
            # Get original prediction
            orig_pred, _ = engine.predict(image)
            
            # Create adversarial example
            adv_image, attack_info = engine.create_adversarial_example(
                image, orig_pred, method
            )
            
            method_results.append(attack_info)
        
        # Calculate method statistics
        success_rate = sum(1 for r in method_results if r.get('success', False)) / len(method_results)
        avg_perturbation = np.mean([r.get('perturbation_norm', 0) for r in method_results])
        
        benchmark_results[method] = {
            'success_rate': success_rate,
            'avg_perturbation': avg_perturbation,
            'results': method_results
        }
    
    return benchmark_results


if __name__ == "__main__":
    print("Adversarial Attack Utilities")
    print("=" * 30)
    print("This module provides utility functions for adversarial attack analysis.")
    print("Import and use the classes and functions as needed.")
