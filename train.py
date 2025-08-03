"""
Memory-Optimized Training Script for Event Camera Semantic Segmentation
Enhanced with aggressive memory management and monitoring
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
from tqdm import tqdm
import time
import logging
from typing import Dict, Optional
import gc
import psutil

from config import Config, create_config
from model import create_model
from dataloader import create_dataloader
from loss import STCLoss
from eval import EventSegmentationEvaluator


def setup_seed(seed):
    """Setup random seeds for reproducibility"""
    print(f'Random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def clear_memory():
    """Aggressive memory clearing"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_usage():
    """Get current memory usage"""
    # System memory
    system_memory = psutil.virtual_memory()

    # GPU memory
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            'allocated': torch.cuda.memory_allocated() / 1024**3,
            'reserved': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3
        }

    return {
        'system': {
            'used': system_memory.used / 1024**3,
            'available': system_memory.available / 1024**3,
            'total': system_memory.total / 1024**3,
            'percent': system_memory.percent
        },
        'gpu': gpu_memory
    }


class MemoryMonitor:
    """Memory usage monitor"""

    def __init__(self, log_interval=10):
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.step_count = 0

    def step(self):
        """Monitor memory usage"""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            usage = get_memory_usage()

            # Log system memory
            sys_mem = usage['system']
            self.logger.debug(f"System Memory: {sys_mem['used']:.1f}GB/"
                            f"{sys_mem['total']:.1f}GB ({sys_mem['percent']:.1f}%)")

            # Log GPU memory
            if usage['gpu']:
                gpu_mem = usage['gpu']
                self.logger.debug(f"GPU Memory: {gpu_mem['allocated']:.1f}GB allocated, "
                                f"{gpu_mem['reserved']:.1f}GB reserved")

            # Warning for high memory usage
            if sys_mem['percent'] > 80:
                self.logger.warning(f"High system memory usage: {sys_mem['percent']:.1f}%")
                clear_memory()

            if usage['gpu'] and gpu_mem['allocated'] > 6.0:  # More than 6GB
                self.logger.warning(f"High GPU memory usage: {gpu_mem['allocated']:.1f}GB")
                clear_memory()


class Trainer:
    """Memory-optimized trainer class"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # Setup logging
        self.setup_logging()

        # Memory monitor
        self.memory_monitor = MemoryMonitor()

        # Initialize model
        self.logger.info("Initializing model...")
        self.model = create_model(cfg.get_model_config())
        self.model = self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Loss function
        self.criterion = STCLoss(k=cfg.train.k, t=cfg.train.t)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Scheduler
        self.scheduler = self._create_scheduler()

        # Mixed precision training (disabled for memory stability)
        self.scaler = None  # Disabled for memory optimization

        # Best metrics
        self.best_loss = float('inf')
        self.best_iou = 0.0

        # Gradient accumulation
        self.accumulation_steps = cfg.memory.gradient_accumulation_steps

        # Create save directory
        os.makedirs(cfg.train.model_save_root, exist_ok=True)

        # Memory tracking
        self.initial_memory = get_memory_usage()
        self.logger.info(f"Initial memory usage: System {self.initial_memory['system']['used']:.1f}GB")

    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        log_file = os.path.join(self.cfg.log_dir, f'train_{time.strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _create_optimizer(self):
        """Create optimizer"""
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.cfg.train.optimizer == 'adam':
            return optim.Adam(params, lr=self.cfg.train.learning_rate,
                            betas=self.cfg.train.betas, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.optimizer == 'adamw':
            return optim.AdamW(params, lr=self.cfg.train.learning_rate,
                             betas=self.cfg.train.betas, weight_decay=self.cfg.train.weight_decay)
        elif self.cfg.train.optimizer == 'sgd':
            return optim.SGD(params, lr=self.cfg.train.learning_rate,
                           momentum=self.cfg.train.momentum, weight_decay=self.cfg.train.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.train.optimizer}")

    def _create_scheduler(self):
        """Create learning rate scheduler"""
        if self.cfg.train.scheduler == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer,
                                           step_size=self.cfg.train.step_size,
                                           gamma=self.cfg.train.gamma)
        elif self.cfg.train.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                       T_max=self.cfg.train.epochs)
        elif self.cfg.train.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                       mode='min', patience=5)
        else:
            return None

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with memory monitoring"""
        self.model.train()

        epoch_loss = 0.0
        num_batches = 0
        successful_batches = 0

        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            try:
                # Memory monitoring
                self.memory_monitor.step()

                # Forward pass
                loss = self.train_step(batch, batch_idx)

                if loss is not None:
                    # Update metrics
                    epoch_loss += loss
                    successful_batches += 1

                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                        'mem': f'{get_memory_usage()["system"]["percent"]:.1f}%'
                    })

                    # Save best model based on loss
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.save_checkpoint('best_loss.pth', epoch, loss)

                num_batches += 1

                # Clear cache more frequently
                if batch_idx % self.cfg.memory.clear_cache_frequency == 0:
                    clear_memory()

            except Exception as e:
                self.logger.error(f"Error in batch {batch_idx}: {e}")
                clear_memory()  # Clear memory on error
                continue

        pbar.close()

        # Update scheduler
        avg_loss = epoch_loss / max(successful_batches, 1)
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

        self.logger.info(f"Epoch {epoch}: Processed {successful_batches}/{num_batches} batches successfully")

        return avg_loss

    def train_step(self, batch, batch_idx):
        """Single training step with memory optimization"""
        try:
            # Extract data
            voxel_ev = batch['voxel_ev']
            seg_label = batch['seg_label'].float().to(self.device)
            p2v_map = batch['p2v_map'].long().to(self.device)

            # Forward pass
            preds, voxel = self.model(voxel_ev)

            # Calculate loss
            loss = self.criterion(voxel, p2v_map, preds, seg_label)
            loss = loss / self.accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.cfg.train.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.gradient_clip)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

            return loss.item() * self.accumulation_steps

        except Exception as e:
            self.logger.error(f"Error in train_step: {e}")
            # Clear gradients on error
            self.optimizer.zero_grad()
            return None

    def validate(self, val_loader, epoch):
        """Validation with memory optimization"""
        self.model.eval()
        evaluator = EventSegmentationEvaluator(self.cfg.eval)

        total_loss = 0.0
        num_batches = 0
        successful_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                if batch is None:
                    continue

                try:
                    # Extract data
                    voxel_ev = batch['voxel_ev']
                    seg_label = batch['seg_label'].float().to(self.device)
                    p2v_map = batch['p2v_map'].long().to(self.device)

                    # Forward pass
                    preds, voxel = self.model(voxel_ev)

                    # Calculate loss
                    loss = self.criterion(voxel, p2v_map, preds, seg_label)
                    total_loss += loss.item()

                    # Map predictions to points for evaluation
                    try:
                        # Ensure valid mapping
                        valid_mask = (p2v_map >= 0) & (p2v_map < len(preds))
                        if valid_mask.sum() > 0:
                            point_preds = preds[p2v_map[valid_mask]].squeeze().cpu()
                            point_labels = seg_label[valid_mask].cpu()
                        else:
                            # Fallback to direct comparison
                            min_len = min(len(preds), len(seg_label))
                            point_preds = preds[:min_len].squeeze().cpu()
                            point_labels = seg_label[:min_len].cpu()

                        # Update evaluator
                        evaluator.update(point_preds, point_labels, batch_idx)
                        successful_batches += 1

                    except Exception as e:
                        self.logger.warning(f"Error in evaluation update: {e}")

                except Exception as e:
                    self.logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
                finally:
                    num_batches += 1

                    # More frequent memory clearing during validation
                    if batch_idx % 2 == 0:
                        clear_memory()

        # Calculate metrics
        avg_loss = total_loss / max(successful_batches, 1)

        try:
            iou = evaluator.evaluate_semantic_segmentation_miou(thresh=self.cfg.eval.default_threshold)
            accuracy = evaluator.evaluate_semantic_segmentation_accuracy(thresh=self.cfg.eval.default_threshold)
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            iou = torch.tensor(0.0)
            accuracy = torch.tensor(0.0)

        # Save best IoU model
        if iou > self.best_iou:
            self.best_iou = iou
            self.save_checkpoint('best_iou.pth', epoch, avg_loss, iou)

        self.logger.info(f"Validation: {successful_batches}/{num_batches} batches processed successfully")

        return avg_loss, iou, accuracy

    def save_checkpoint(self, filename, epoch, loss, iou=None):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'iou': iou,
            'config': self.cfg
        }

        path = os.path.join(self.cfg.train.model_save_root, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

    def train(self, train_loader, val_loader, resume_path=None):
        """Main training loop with memory optimization"""
        start_epoch = 0

        # Resume training
        if resume_path and os.path.exists(resume_path):
            self.logger.info(f"Resuming from {resume_path}")
            start_epoch = self.load_checkpoint(resume_path)

        self.logger.info(f"Starting training for {self.cfg.train.epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.cfg.data.batch_size}")
        self.logger.info(f"Accumulation steps: {self.accumulation_steps}")
        self.logger.info(f"Memory monitoring enabled: {self.cfg.memory.enable_memory_monitoring}")

        for epoch in range(start_epoch, self.cfg.train.epochs):
            epoch_start = time.time()

            # Training
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation (start after warmup to save memory)
            if epoch >= 5:  # Start validation after warmup
                try:
                    val_loss, iou, accuracy = self.validate(val_loader, epoch)
                    self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                                   f"Val Loss={val_loss:.4f}, IoU={iou:.4f}, Acc={accuracy:.4f}")
                except Exception as e:
                    self.logger.error(f"Validation failed: {e}")
                    self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")
            else:
                self.logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.cfg.train.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, train_loss)

            # Memory status
            current_memory = get_memory_usage()
            self.logger.info(f"Memory: System {current_memory['system']['percent']:.1f}%")

            epoch_time = time.time() - epoch_start
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

            # Aggressive cleanup after each epoch
            clear_memory()

        self.logger.info("Training completed!")
        self.logger.info(f"Best Loss: {self.best_loss:.4f}, Best IoU: {self.best_iou:.4f}")


def main():
    """Main training function"""
    # Create configuration
    cfg = create_config()

    # Setup seed
    setup_seed(cfg.seed)

    # Create data loaders
    print("Creating data loaders...")
    try:
        train_loader = create_dataloader(cfg, mode='train')
        val_loader = create_dataloader(cfg, mode='val')

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    # Create trainer
    trainer = Trainer(cfg)

    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
    