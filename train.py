"""
Main training script for FastFlow anomaly detection

Usage:
    python train.py --data_path /path/to/mvtec --category bottle
"""

import argparse
import torch
from pathlib import Path
import json

# Import custom modules
from model import FastFlowModel
from dataset import MVTecDataModule
from trainer import FastFlowTrainer, FastFlowEvaluator


class FastFlowPipeline:
    """Complete pipeline for FastFlow training and evaluation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.data_module = None
        self.trainer = None
        self.evaluator = None
    
    def setup(self):
        """Setup model and data"""
        print("\n=== Setting up FastFlow Pipeline ===")
        
        # Initialize model
        print(f"\nInitializing FastFlow model with backbone: {self.config['backbone']}")
        self.model = FastFlowModel(
            backbone_name=self.config["backbone"],
            flow_steps=self.config["flow_steps"],
            input_size=tuple(self.config["image_size"]),
            hidden_ratio=self.config.get("hidden_ratio", 1.0),
            clamp=self.config.get("clamp", 2.0),
            conv3x3_only=self.config.get("conv3x3_only", False),
        )
        
        # Initialize data module
        print(f"Loading MVTec dataset - Category: {self.config['category']}")
        self.data_module = MVTecDataModule(
            root_dir=self.config['data_path'],
            category=self.config['category'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            image_size=tuple(self.config['image_size'])
        )
        self.data_module.setup()
        
        # Initialize trainer
        save_dir = Path(self.config['save_dir']) / self.config['category']
        self.trainer = FastFlowTrainer(
            model=self.model,
            device=str(self.device),
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay'],
            save_dir=str(save_dir)
        )
        
        print("\n✓ Setup completed successfully")
    
    def train(self):
        """Train the model"""
        print("\n=== Starting Training ===")
        
        train_loader = self.data_module.train_dataloader()
        val_loader = self.data_module.test_dataloader()
        
        self.trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['num_epochs'],
            patience=self.config['patience']
        )
    
    def evaluate(self):
        """Evaluate the model"""
        print("\n=== Evaluating Model ===")
        
        # Load best model
        self.trainer.load_checkpoint('best_model.pth')
        
        # Initialize evaluator
        self.evaluator = FastFlowEvaluator(
            model=self.model,
            device=str(self.device)
        )
        
        # Get predictions
        test_loader = self.data_module.test_dataloader()
        predictions = self.evaluator.predict(test_loader)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(predictions)
        
        print("\n=== Evaluation Results ===")
        print(f"Image-level AUROC: {metrics['image_auroc']:.4f}")
        print(f"Pixel-level AUROC: {metrics['pixel_auroc']:.4f}")
        
        # Save metrics
        save_dir = Path(self.config['save_dir']) / self.config['category']
        metrics_path = save_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"\nMetrics saved to {metrics_path}")
        
        # Visualize results
        results_dir = save_dir / 'visualizations'
        self.evaluator.visualize_results(
            test_loader,
            save_dir=str(results_dir),
            num_samples=self.config['num_visualizations']
        )
        
        return metrics
    
    def run(self):
        """Run complete pipeline"""
        self.setup()
        self.train()
        metrics = self.evaluate()
        return metrics


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train FastFlow on MVTec AD')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to MVTec AD dataset')
    parser.add_argument('--category', type=str, required=True,
                        help='Product category to train on')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'wide_resnet50_2'],
                        help='Backbone architecture')
    parser.add_argument('--flow_steps', type=int, default=8,
                        help='Number of flow steps')
    parser.add_argument('--hidden_ratio', type=float, default=1.0,
                    help='Coupling subnet hidden ratio')
    parser.add_argument('--clamp', type=float, default=2.0,
                        help='Clamp for coupling scale')
    parser.add_argument('--conv3x3_only', action='store_true',
                        help='Use only 3x3 kernels in coupling steps')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Other arguments
    parser.add_argument('--image_size', type=int, nargs=2, default=[416, 416],
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--num_visualizations', type=int, default=10,
                        help='Number of samples to visualize')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Convert to config dict
    config = vars(args)
    
    print("=== FastFlow Anomaly Detection ===")
    print(f"\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Run pipeline
    pipeline = FastFlowPipeline(config)
    metrics = pipeline.run()
    
    print("\n=== Training and Evaluation Completed ===")
    print(f"Final Image AUROC: {metrics['image_auroc']:.4f}")
    print(f"Final Pixel AUROC: {metrics['pixel_auroc']:.4f}")


if __name__ == '__main__':
    main()