import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

class TrainingVisualizer:
    def __init__(self, log_file: str, save_dir: str = 'plots'):
        """Initialize visualizer with log file path and save directory"""
        self.log_file = log_file
        self.save_dir = save_dir
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        os.makedirs(save_dir, exist_ok=True)
        
    def parse_log_file(self) -> None:
        """Parse the log file to extract metrics"""
        try:
            current_epoch = 0
            epoch_metrics = {
                'train_loss': None,
                'train_acc': None,
                'val_loss': None,
                'val_acc': None
            }
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    # Extract training metrics
                    if "Training Epoch" in line and "Results" in line:
                        current_epoch = int(re.search(r'Epoch (\d+)', line).group(1))
                    elif "Loss:" in line:
                        if "Training Epoch" in line:
                            epoch_metrics['train_loss'] = float(re.search(r'Loss: ([\d.]+)', line).group(1))
                    elif "Accuracy:" in line:
                        if "Training Epoch" in line:
                            epoch_metrics['train_acc'] = float(re.search(r'Accuracy: ([\d.]+)', line).group(1))
                            
                    # Extract validation metrics
                    elif "Validation Epoch" in line and "Results" in line:
                        pass  # Skip the header line
                    elif "Loss:" in line and "Validation" in line:
                        epoch_metrics['val_loss'] = float(re.search(r'Loss: ([\d.]+)', line).group(1))
                    elif "Accuracy:" in line and "Validation" in line:
                        epoch_metrics['val_acc'] = float(re.search(r'Accuracy: ([\d.]+)', line).group(1))
                        
                        # After getting validation accuracy, we have all metrics for this epoch
                        if all(v is not None for v in epoch_metrics.values()):
                            self.metrics['epochs'].append(current_epoch)
                            self.metrics['train_loss'].append(epoch_metrics['train_loss'])
                            self.metrics['train_acc'].append(epoch_metrics['train_acc'])
                            self.metrics['val_loss'].append(epoch_metrics['val_loss'])
                            self.metrics['val_acc'].append(epoch_metrics['val_acc'])
                            
                            # Reset metrics for next epoch
                            epoch_metrics = {k: None for k in epoch_metrics}
                            
        except Exception as e:
            print(f"Error parsing log file: {str(e)}")
            raise
            
    def plot_metrics(self) -> None:
        """Create and save visualization plots"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Set style
            plt.style.use('seaborn')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
            
            # Plot Loss
            ax1.plot(self.metrics['epochs'], self.metrics['train_loss'], 
                    label='Training Loss', marker='o')
            ax1.plot(self.metrics['epochs'], self.metrics['val_loss'], 
                    label='Validation Loss', marker='s')
            ax1.set_title('Training and Validation Loss Over Time', pad=20)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot Accuracy
            ax2.plot(self.metrics['epochs'], self.metrics['train_acc'], 
                    label='Training Accuracy', marker='o')
            ax2.plot(self.metrics['epochs'], self.metrics['val_acc'], 
                    label='Validation Accuracy', marker='s')
            ax2.set_title('Training and Validation Accuracy Over Time', pad=20)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            
            # Adjust layout and save
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f'training_metrics_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Plots saved to: {save_path}")
            
            # Save metrics to CSV
            df = pd.DataFrame({
                'epoch': self.metrics['epochs'],
                'train_loss': self.metrics['train_loss'],
                'train_acc': self.metrics['train_acc'],
                'val_loss': self.metrics['val_loss'],
                'val_acc': self.metrics['val_acc']
            })
            csv_path = os.path.join(self.save_dir, f'training_metrics_{timestamp}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved to: {csv_path}")
            
        except Exception as e:
            print(f"Error creating plots: {str(e)}")
            raise
            
    def generate_training_summary(self) -> str:
        """Generate a summary of the training results"""
        try:
            summary = []
            summary.append("Training Summary")
            summary.append("=" * 50)
            
            # Best metrics
            best_train_acc = max(self.metrics['train_acc'])
            best_val_acc = max(self.metrics['val_acc'])
            best_train_loss = min(self.metrics['train_loss'])
            best_val_loss = min(self.metrics['val_loss'])
            
            summary.append(f"Total Epochs: {len(self.metrics['epochs'])}")
            summary.append(f"Best Training Accuracy: {best_train_acc:.4f}")
            summary.append(f"Best Validation Accuracy: {best_val_acc:.4f}")
            summary.append(f"Best Training Loss: {best_train_loss:.4f}")
            summary.append(f"Best Validation Loss: {best_val_loss:.4f}")
            
            # Final metrics
            final_train_acc = self.metrics['train_acc'][-1]
            final_val_acc = self.metrics['val_acc'][-1]
            final_train_loss = self.metrics['train_loss'][-1]
            final_val_loss = self.metrics['val_loss'][-1]
            
            summary.append("\nFinal Metrics:")
            summary.append(f"Final Training Accuracy: {final_train_acc:.4f}")
            summary.append(f"Final Validation Accuracy: {final_val_acc:.4f}")
            summary.append(f"Final Training Loss: {final_train_loss:.4f}")
            summary.append(f"Final Validation Loss: {final_val_loss:.4f}")
            
            # Save summary to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            summary_path = os.path.join(self.save_dir, f'training_summary_{timestamp}.txt')
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary))
                
            print(f"Training summary saved to: {summary_path}")
            return '\n'.join(summary)
            
        except Exception as e:
            print(f"Error generating training summary: {str(e)}")
            raise

def visualize_training(log_file: str, save_dir: str = 'plots') -> None:
    """Convenience function to visualize training results"""
    try:
        visualizer = TrainingVisualizer(log_file, save_dir)
        visualizer.parse_log_file()
        visualizer.plot_metrics()
        summary = visualizer.generate_training_summary()
        print("\nTraining Summary:")
        print(summary)
    except Exception as e:
        print(f"Error visualizing training results: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Visualize training metrics from log file')
    parser.add_argument('log_file', type=str, help='Path to the training log file')
    parser.add_argument('--save_dir', type=str, default='plots', 
                        help='Directory to save plots and metrics')
    args = parser.parse_args()
    
    visualize_training(args.log_file, args.save_dir) 