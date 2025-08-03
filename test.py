"""
Unified Testing/Inference Script for Event Camera Semantic Segmentation
Includes comprehensive evaluation and visualization
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import logging
import json
import cv2
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

from config import Config, create_config
from model import create_model
from dataloader import create_dataloader
from eval import EventSegmentationEvaluator


class Tester:
    """Main testing class with comprehensive evaluation"""
    
    def __init__(self, cfg: Config, checkpoint_path: str):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.logger.info("Initializing model...")
        self.model = create_model(cfg.get_model_config())
        self.model = self.model.to(self.device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Create result directory
        self.result_dir = os.path.join(cfg.result_dir, f'test_{time.strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = EventSegmentationEvaluator(cfg.eval)
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        log_file = os.path.join(self.cfg.log_dir, f'test_{time.strftime("%Y%m%d_%H%M%S")}.log')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        self.logger.info(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel wrapped models
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.logger.info('Checkpoint loaded successfully')
        
        # Log checkpoint info if available
        if 'epoch' in checkpoint:
            self.logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            self.logger.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
        if 'iou' in checkpoint:
            self.logger.info(f"Checkpoint IoU: {checkpoint['iou']:.4f}")
    
    def test_single_batch(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Test on a single batch"""
        with torch.no_grad():
            # Extract data
            voxel_ev = batch['voxel_ev']
            labels = batch['seg_label'].to(self.device).float()
            p2v_map = batch['p2v_map'].to(self.device).long()
            locs = batch['locs']
            
            # Forward pass
            start_time = time.time()
            output, voxel_output = self.model(voxel_ev)
            inference_time = time.time() - start_time
            
            # Process predictions
            predictions = output.squeeze()
            
            # Map voxel predictions to points for evaluation
            point_predictions = predictions[p2v_map].squeeze()
            
            # Calculate metrics for this batch
            batch_metrics = {
                'inference_time': inference_time,
                'num_events': locs.shape[0],
                'num_voxels': len(predictions)
            }
            
            return point_predictions, labels, batch_metrics
    
    def visualize_results(self, locs: np.ndarray, predictions: np.ndarray,
                         labels: np.ndarray, save_path: str, threshold: float = 0.5):
        """Visualize segmentation results"""
        # Event camera resolution
        h, w = 260, 346
        
        # Create visualization frames
        gt_frame = np.zeros((h, w, 3), dtype=np.uint8)
        pred_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map events to pixels
        for i in range(locs.shape[0]):
            x = int(locs[i, 1])  # x coordinate
            y = int(locs[i, 2])  # y coordinate
            
            if 0 <= x < w and 0 <= y < h:
                # Ground truth (green for positive)
                if labels[i] > 0:
                    gt_frame[y, x] = [0, 255, 0]
                else:
                    gt_frame[y, x] = [100, 100, 100]
                
                # Predictions (red for positive)
                if predictions[i] > threshold:
                    pred_frame[y, x] = [0, 0, 255]
                else:
                    pred_frame[y, x] = [100, 100, 100]
        
        # Combine frames
        comparison = np.hstack([gt_frame, pred_frame])
        
        # Add labels
        cv2.putText(comparison, 'Ground Truth', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, 'Predictions', (w + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save visualization
        cv2.imwrite(save_path, comparison)
    
    def save_predictions(self, predictions: torch.Tensor, locs: torch.Tensor,
                        batch_idx: int, threshold: float = 0.5):
        """Save predictions in various formats"""
        pred_np = predictions.cpu().numpy()
        locs_np = locs.cpu().numpy()
        
        # Save raw predictions
        raw_path = os.path.join(self.result_dir, f'predictions_batch_{batch_idx}.npz')
        np.savez(raw_path, predictions=pred_np, locations=locs_np)
        
        # Save binary predictions
        binary_pred = (pred_np > threshold).astype(np.uint8)
        binary_path = os.path.join(self.result_dir, f'binary_predictions_batch_{batch_idx}.npy')
        np.save(binary_path, binary_pred)
        
        # Save as point cloud (for visualization)
        positive_mask = binary_pred > 0
        positive_locs = locs_np[positive_mask]
        
        if positive_locs.shape[0] > 0:
            pcd_path = os.path.join(self.result_dir, f'positive_points_batch_{batch_idx}.txt')
            np.savetxt(pcd_path, positive_locs[:, 1:4], fmt='%.6f')  # Save x, y, t coordinates
    
    def test(self, test_loader: DataLoader):
        """Main testing loop"""
        self.logger.info('Starting testing...')
        self.logger.info(f'Number of test batches: {len(test_loader)}')
        
        all_predictions = []
        all_labels = []
        total_inference_time = 0
        total_events = 0
        total_voxels = 0
        
        # Testing loop
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Testing')):
            if batch is None:
                continue
            
            try:
                # Test single batch
                predictions, labels, batch_metrics = self.test_single_batch(batch)
                
                # Store for evaluation
                self.evaluator.update(predictions, labels, batch_idx)
                
                all_predictions.append(predictions)
                all_labels.append(labels)
                
                # Update statistics
                total_inference_time += batch_metrics['inference_time']
                total_events += batch_metrics['num_events']
                total_voxels += batch_metrics['num_voxels']
                
                # Save predictions for first few batches
                if batch_idx < 10:
                    self.save_predictions(predictions, batch['locs'], batch_idx)
                
                # Visualize results for first few batches
                if batch_idx < 5:
                    vis_path = os.path.join(self.result_dir, f'visualization_batch_{batch_idx}.png')
                    self.visualize_results(
                        batch['locs'].cpu().numpy(),
                        predictions.cpu().numpy(),
                        labels.cpu().numpy(),
                        vis_path
                    )
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Calculate overall metrics
        self.logger.info('Calculating metrics...')
        
        results = {}
        
        # Evaluate at different thresholds
        df_metrics = self.evaluator.evaluate_all_thresholds(self.cfg.eval.confidence_thresholds)
        
        # Find best threshold
        best_idx = df_metrics['miou'].argmax()
        best_threshold = df_metrics.iloc[best_idx]['threshold']
        
        results['best_threshold'] = best_threshold
        results['metrics_at_best'] = {
            'miou': df_metrics.iloc[best_idx]['miou'],
            'accuracy': df_metrics.iloc[best_idx]['accuracy'],
            'precision': df_metrics.iloc[best_idx]['precision'],
            'recall': df_metrics.iloc[best_idx]['recall'],
            'f1': df_metrics.iloc[best_idx]['f1']
        }
        
        # Default threshold metrics
        default_idx = (df_metrics['threshold'] - self.cfg.eval.default_threshold).abs().argmin()
        results['metrics_at_default'] = {
            'threshold': self.cfg.eval.default_threshold,
            'miou': df_metrics.iloc[default_idx]['miou'],
            'accuracy': df_metrics.iloc[default_idx]['accuracy'],
            'precision': df_metrics.iloc[default_idx]['precision'],
            'recall': df_metrics.iloc[default_idx]['recall'],
            'f1': df_metrics.iloc[default_idx]['f1']
        }
        
        # Log results
        self.logger.info(f"Best threshold: {best_threshold:.3f}")
        self.logger.info(f"Best mIoU: {results['metrics_at_best']['miou']:.4f}")
        self.logger.info(f"Best accuracy: {results['metrics_at_best']['accuracy']:.4f}")
        
        # Performance statistics
        avg_inference_time = total_inference_time / len(test_loader)
        events_per_second = total_events / total_inference_time if total_inference_time > 0 else 0
        
        results['performance'] = {
            'avg_inference_time_per_batch': avg_inference_time,
            'events_per_second': events_per_second,
            'total_events': total_events,
            'total_voxels': total_voxels
        }
        
        self.logger.info(f'Average inference time per batch: {avg_inference_time:.4f}s')
        self.logger.info(f'Processing speed: {events_per_second:.0f} events/second')
        
        # Generate plots
        self.logger.info('Generating evaluation plots...')
        self.evaluator.plot_metrics(self.result_dir)
        
        # Generate report
        report_path = os.path.join(self.result_dir, 'evaluation_report.txt')
        report = self.evaluator.generate_report(report_path)
        self.logger.info(f'Evaluation report saved to {report_path}')
        
        # Save results JSON
        results_path = os.path.join(self.result_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        self.logger.info(f'Results saved to {results_path}')
        
        return results
    
    def test_single_sequence(self, event_data: Dict) -> np.ndarray:
        """Test on a single event sequence (for real-time inference)"""
        self.model.eval()
        
        with torch.no_grad():
            # Prepare input (assuming pre-voxelized data)
            voxel_ev = event_data['voxel_ev']
            
            # Forward pass
            output, _ = self.model(voxel_ev)
            predictions = output.squeeze().cpu().numpy()
            
            return predictions


def main():
    """Main testing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Event Camera Semantic Segmentation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--threshold', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--save_predictions', action='store_true', help='Save predictions')
    
    args = parser.parse_args()
    
    # Create configuration
    cfg = create_config(args.config)
    
    if args.threshold is not None:
        cfg.eval.default_threshold = args.threshold
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader = create_dataloader(cfg, mode='test')
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create tester
    tester = Tester(cfg, args.checkpoint)
    
    # Run testing
    results = tester.test(test_loader)
    
    print("\nTesting completed!")
    print(f"Results saved to: {tester.result_dir}")


if __name__ == '__main__':
    main()