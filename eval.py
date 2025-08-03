"""
Unified Evaluation Module for Event Camera Semantic Segmentation
Comprehensive metrics including IoU, accuracy, precision, recall, and ROC analysis
"""
import os
import torch
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


class EventSegmentationEvaluator:
    """Comprehensive evaluator for event-based segmentation"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.matches = {}
        self.data = pd.DataFrame()
        
        # ROC evaluation parameters
        if hasattr(cfg, 'roc') and cfg.roc:
            self.pd_detT = cfg.pd_detT
            self.correct_thresh = cfg.correct_thresh
            self.whole_ev_num = 0
            self.obj_num = 0
            self.frame_num = 0
            self.correct_num = 0
            self.false_num = 0
        
        # Additional metrics
        self.confusion_matrices = {}
        self.per_class_metrics = {}
        self.threshold_metrics = {}
    
    def update(self, predictions: torch.Tensor, labels: torch.Tensor,
               batch_idx: int, additional_info: Optional[Dict] = None):
        """Update evaluator with batch predictions"""
        # Ensure consistent shapes
        if predictions.dim() > 1:
            predictions = predictions.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Handle shape mismatch
        min_len = min(predictions.shape[0], labels.shape[0])
        predictions = predictions[:min_len]
        labels = labels[:min_len]
        
        self.matches[batch_idx] = {
            'seg_pred': predictions.detach().cpu(),
            'seg_gt': labels.detach().cpu()
        }
        
        if additional_info:
            self.matches[batch_idx].update(additional_info)
    
    def evaluate_semantic_segmentation_miou(self, thresh: float = 0.5) -> torch.Tensor:
        """Calculate mean IoU for binary segmentation"""
        seg_gt_list = []
        seg_pred_list = []
        
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        
        if not seg_gt_list:
            return torch.tensor(0.0)
        
        seg_gt_all = torch.cat(seg_gt_list, dim=0)
        seg_pred_all = torch.cat(seg_pred_list, dim=0)
        
        # Ensure consistent shapes
        min_len = min(seg_gt_all.shape[0], seg_pred_all.shape[0])
        seg_gt_all = seg_gt_all[:min_len]
        seg_pred_all = seg_pred_all[:min_len]
        
        # Binarize predictions
        seg_pred_binary = (seg_pred_all >= thresh).float()
        
        # Calculate IoU
        intersection = ((seg_gt_all == 1) & (seg_pred_binary == 1)).sum().float()
        union = ((seg_gt_all == 1) | (seg_pred_binary == 1)).sum().float()
        
        iou = intersection / union if union > 0 else torch.tensor(0.0)
        
        return iou
    
    def evaluate_semantic_segmentation_accuracy(self, thresh: float = 0.5) -> torch.Tensor:
        """Calculate pixel-wise accuracy"""
        seg_gt_list = []
        seg_pred_list = []
        
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        
        if not seg_gt_list:
            return torch.tensor(0.0)
        
        seg_gt_all = torch.cat(seg_gt_list, dim=0)
        seg_pred_all = torch.cat(seg_pred_list, dim=0)
        
        # Ensure consistent shapes
        min_len = min(seg_gt_all.shape[0], seg_pred_all.shape[0])
        seg_gt_all = seg_gt_all[:min_len]
        seg_pred_all = seg_pred_all[:min_len]
        
        # Binarize predictions
        seg_pred_binary = (seg_pred_all >= thresh).float()
        
        # Calculate accuracy
        correct = (seg_gt_all == seg_pred_binary).sum().float()
        total = seg_gt_all.shape[0]
        accuracy = correct / total if total > 0 else torch.tensor(0.0)
        
        return accuracy
    
    def evaluate_precision_recall_f1(self, thresh: float = 0.5) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        seg_gt_list = []
        seg_pred_list = []
        
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        
        if not seg_gt_list:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        seg_gt_all = torch.cat(seg_gt_list, dim=0).numpy()
        seg_pred_all = torch.cat(seg_pred_list, dim=0).numpy()
        
        # Ensure consistent shapes
        min_len = min(seg_gt_all.shape[0], seg_pred_all.shape[0])
        seg_gt_all = seg_gt_all[:min_len]
        seg_pred_all = seg_pred_all[:min_len]
        
        # Binarize predictions
        seg_pred_binary = (seg_pred_all >= thresh).astype(int)
        
        # Calculate metrics
        tp = np.sum((seg_pred_binary == 1) & (seg_gt_all == 1))
        fp = np.sum((seg_pred_binary == 1) & (seg_gt_all == 0))
        fn = np.sum((seg_pred_binary == 0) & (seg_gt_all == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_all_thresholds(self, thresholds: Optional[List[float]] = None) -> pd.DataFrame:
        """Evaluate metrics across multiple thresholds"""
        if thresholds is None:
            thresholds = getattr(self.cfg, 'confidence_thresholds', [0.1, 0.3, 0.5, 0.7, 0.9])
        
        results = []
        
        for thresh in tqdm(thresholds, desc='Evaluating thresholds'):
            miou = self.evaluate_semantic_segmentation_miou(thresh)
            accuracy = self.evaluate_semantic_segmentation_accuracy(thresh)
            pr_metrics = self.evaluate_precision_recall_f1(thresh)
            
            results.append({
                'threshold': thresh,
                'miou': miou.item(),
                'accuracy': accuracy.item(),
                'precision': pr_metrics['precision'],
                'recall': pr_metrics['recall'],
                'f1': pr_metrics['f1']
            })
        
        return pd.DataFrame(results)
    
    def generate_confusion_matrix(self, thresh: float = 0.5) -> np.ndarray:
        """Generate confusion matrix"""
        seg_gt_list = []
        seg_pred_list = []
        
        for k, v in self.matches.items():
            seg_gt_list.append(v['seg_gt'])
            seg_pred_list.append(v['seg_pred'])
        
        if not seg_gt_list:
            return np.zeros((2, 2))
        
        seg_gt_all = torch.cat(seg_gt_list, dim=0).numpy().flatten()
        seg_pred_all = torch.cat(seg_pred_list, dim=0).numpy().flatten()
        
        # Ensure consistent shapes
        min_len = min(seg_gt_all.shape[0], seg_pred_all.shape[0])
        seg_gt_all = seg_gt_all[:min_len]
        seg_pred_all = seg_pred_all[:min_len]
        
        # Binarize predictions
        seg_pred_binary = (seg_pred_all >= thresh).astype(int)
        
        # Generate confusion matrix
        cm = confusion_matrix(seg_gt_all, seg_pred_binary, labels=[0, 1])
        return cm
    
    def roc_update(self, ts: np.ndarray, preds: np.ndarray, idx: np.ndarray,
                   label: np.ndarray, ev_locs: np.ndarray, thresh: float = 0.9):
        """Update ROC evaluation metrics"""
        self.whole_ev_num += preds.shape[0]
        self.frame_num += int((ts.max() - ts.min()) / self.pd_detT)
        
        for i in range(int((ts.max() - ts.min()) / self.pd_detT + 1)):
            t_range = (ts > i * self.pd_detT) * (ts < (i + 1) * self.pd_detT)
            idx_frame = idx[t_range]
            preds_frame = preds[t_range]
            label_frame = label[t_range]
            ev_locs_frame = ev_locs[t_range, 1:4]
            
            preds_frame_ori = preds_frame.copy()
            idx_list_frame = set(idx_frame)
            false_mask = np.zeros((260, 346), dtype=np.uint8)
            
            # Binarize predictions
            preds_frame_binary = (preds_frame_ori >= thresh).astype(int)
            
            # Calculate detection Pd
            for idx_i in idx_list_frame:
                if idx_i != 0:  # Object
                    self.obj_num += 1
                    mask = idx_frame == idx_i
                    preds_frame_i = preds_frame_binary[mask]
                    label_frame_i = label_frame[mask]
                    
                    if label_frame_i.sum() > 0:
                        num_correct_frame = (preds_frame_i == label_frame_i).sum()
                        accuracy = num_correct_frame / label_frame_i.sum()
                        
                        if accuracy >= self.correct_thresh:
                            self.correct_num += 1
            
            # Calculate false alarm Fa
            false_mask_indices = (label_frame == 0) * (preds_frame_binary == 1)
            false_ev = ev_locs_frame[false_mask_indices]
            
            for ii in range(false_ev.shape[0]):
                x, y = int(false_ev[ii, 0]), int(false_ev[ii, 1])
                if 0 <= x < 346 and 0 <= y < 260:
                    false_mask[y, x] = 1
            
            # Count connected components as false alarms
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                false_mask, connectivity=8, ltype=cv2.CV_32S
            )
            self.false_num += (num_labels - 1)
    
    def cal_roc(self) -> Tuple[float, float]:
        """Calculate ROC metrics (Pd and Fa)"""
        pd = self.correct_num / self.obj_num if self.obj_num > 0 else 0
        fa = self.false_num / (self.frame_num * 346 * 260) if self.frame_num > 0 else 0
        return pd, fa
    
    def plot_metrics(self, save_dir: str):
        """Plot various evaluation metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # 1. Threshold vs Metrics plot
            df_metrics = self.evaluate_all_thresholds()
            
            plt.figure(figsize=(10, 6))
            for metric in ['miou', 'accuracy', 'precision', 'recall', 'f1']:
                if metric in df_metrics.columns:
                    plt.plot(df_metrics['threshold'], df_metrics[metric], label=metric, marker='o')
            
            plt.xlabel('Threshold')
            plt.ylabel('Score')
            plt.title('Metrics vs Threshold')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'metrics_vs_threshold.png'))
            plt.close()
            
            # 2. Confusion Matrix
            default_thresh = getattr(self.cfg, 'default_threshold', 0.5)
            cm = self.generate_confusion_matrix(default_thresh)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title(f'Confusion Matrix (Threshold={default_thresh})')
            plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
            plt.close()
            
            # 3. Precision-Recall Curve
            precisions = []
            recalls = []
            
            for thresh in np.linspace(0, 1, 101):
                pr_metrics = self.evaluate_precision_recall_f1(thresh)
                precisions.append(pr_metrics['precision'])
                recalls.append(pr_metrics['recall'])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recalls, precisions, 'b-', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.grid(True)
            plt.savefig(os.path.join(save_dir, 'precision_recall_curve.png'))
            plt.close()
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
    
    def generate_report(self, save_path: str):
        """Generate comprehensive evaluation report"""
        report = []
        report.append("=" * 60)
        report.append("Event Camera Semantic Segmentation Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        try:
            # Best threshold based on mIoU
            df_metrics = self.evaluate_all_thresholds()
            if not df_metrics.empty:
                best_idx = df_metrics['miou'].argmax()
                best_thresh = df_metrics.iloc[best_idx]['threshold']
                
                report.append(f"Best Threshold (based on mIoU): {best_thresh:.3f}")
                report.append("")
                
                # Metrics at best threshold
                report.append("Metrics at Best Threshold:")
                report.append("-" * 30)
                for metric in ['miou', 'accuracy', 'precision', 'recall', 'f1']:
                    if metric in df_metrics.columns:
                        value = df_metrics.iloc[best_idx][metric]
                        report.append(f"{metric.upper():>10}: {value:.4f}")
                report.append("")
                
                # Metrics at default threshold
                default_thresh = getattr(self.cfg, 'default_threshold', 0.5)
                default_idx = (df_metrics['threshold'] - default_thresh).abs().argmin()
                report.append(f"Metrics at Default Threshold ({default_thresh}):")
                report.append("-" * 30)
                for metric in ['miou', 'accuracy', 'precision', 'recall', 'f1']:
                    if metric in df_metrics.columns:
                        value = df_metrics.iloc[default_idx][metric]
                        report.append(f"{metric.upper():>10}: {value:.4f}")
                report.append("")
            
            # ROC metrics if available
            if hasattr(self.cfg, 'roc') and self.cfg.roc and self.obj_num > 0:
                pd_rate, fa_rate = self.cal_roc()
                report.append("ROC Metrics:")
                report.append("-" * 30)
                report.append(f"Detection Rate (Pd): {pd_rate:.4f}")
                report.append(f"False Alarm Rate (Fa): {fa_rate:.6f}")
                report.append("")
            
            # Confusion Matrix
            default_thresh = getattr(self.cfg, 'default_threshold', 0.5)
            cm = self.generate_confusion_matrix(default_thresh)
            report.append("Confusion Matrix:")
            report.append("-" * 30)
            report.append(f"True Negatives:  {cm[0, 0]:>10}")
            report.append(f"False Positives: {cm[0, 1]:>10}")
            report.append(f"False Negatives: {cm[1, 0]:>10}")
            report.append(f"True Positives:  {cm[1, 1]:>10}")
            report.append("")
            
            # Save report
            with open(save_path, 'w') as f:
                f.write('\n'.join(report))
            
            # Also save metrics DataFrame
            if not df_metrics.empty:
                csv_path = save_path.replace('.txt', '_metrics.csv')
                df_metrics.to_csv(csv_path, index=False)
            
        except Exception as e:
            report.append(f"Error generating detailed report: {e}")
            with open(save_path, 'w') as f:
                f.write('\n'.join(report))
        
        return '\n'.join(report)
    
    def reset(self):
        """Reset evaluator state"""
        self.matches = {}
        self.data = pd.DataFrame()
        
        if hasattr(self.cfg, 'roc') and self.cfg.roc:
            self.whole_ev_num = 0
            self.obj_num = 0
            self.frame_num = 0
            self.correct_num = 0
            self.false_num = 0
        
        self.confusion_matrices = {}
        self.per_class_metrics = {}
        self.threshold_metrics = {}