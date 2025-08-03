"""
Main Entry Point for Event Camera Semantic Segmentation
Provides unified interface for cache precomputation, training, and testing
"""
import os
import sys
import argparse
import logging
import torch
import gc
from tqdm import tqdm

# Import project modules
from config import create_config, create_memory_optimized_config
from dataloader import EventDataset, create_dataloader
from train import Trainer, setup_seed, clear_memory
from test import Tester


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'main.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def precompute_cache(config):
    """Precompute voxelization cache for all datasets"""
    logger = logging.getLogger(__name__)
    logger.info("Starting cache precomputation...")

    # Check data directories
    modes = ['train', 'val', 'test']
    for mode in modes:
        data_dir = os.path.join(config.data.data_root, mode)
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist, skipping {mode}")
            continue

        logger.info(f"Precomputing cache for {mode} set...")

        try:
            # Create dataset with caching enabled
            dataset = EventDataset(config, mode=mode, use_cache=True)
            logger.info(f"Cache precomputation completed for {mode} set ({len(dataset)} samples)")

            # Clear memory after each dataset
            del dataset
            clear_memory()

        except Exception as e:
            logger.error(f"Error precomputing cache for {mode}: {e}")
            continue

    logger.info("Cache precomputation completed for all datasets")


def train_model(config, resume_path=None):
    """Train the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting training...")

    # Setup seed
    setup_seed(config.seed)

    # Create data loaders
    logger.info("Creating data loaders...")
    try:
        train_loader = create_dataloader(config, mode='train')
        val_loader = create_dataloader(config, mode='val')

        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")

    except Exception as e:
        logger.error(f"Error creating data loaders: {e}")
        return

    # Create trainer
    try:
        trainer = Trainer(config)

        # Start training
        trainer.train(train_loader, val_loader, resume_path)

    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


def test_model(config, checkpoint_path):
    """Test the model"""
    logger = logging.getLogger(__name__)
    logger.info("Starting testing...")

    # Check checkpoint exists
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    # Create test data loader
    try:
        test_loader = create_dataloader(config, mode='test')
        logger.info(f"Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        logger.error(f"Error creating test data loader: {e}")
        return

    # Create tester and run evaluation
    try:
        tester = Tester(config, checkpoint_path)
        results = tester.test(test_loader)

        logger.info("Testing completed successfully!")
        logger.info(f"Results saved to: {tester.result_dir}")

        return results

    except Exception as e:
        logger.error(f"Testing error: {e}")
        raise


def check_environment():
    """Check system environment and requirements"""
    logger = logging.getLogger(__name__)

    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        logger.warning("CUDA not available, using CPU")

    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total / 1024**3:.1f}GB total, "
                   f"{memory.available / 1024**3:.1f}GB available")

        if memory.available < 8 * 1024**3:  # Less than 8GB available
            logger.warning("Low system memory detected, consider using memory-optimized config")

    except ImportError:
        logger.warning("psutil not available, cannot check system memory")


def main():
    parser = argparse.ArgumentParser(description='Event Camera Semantic Segmentation')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['cache', 'train', 'test', 'all'],
                       help='Operation mode')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (for testing or resume training)')
    parser.add_argument('--memory-optimized', action='store_true',
                       help='Use memory-optimized configuration')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--exp-name', type=str, default=None,
                       help='Experiment name')

    args = parser.parse_args()

    # Create configuration
    if args.memory_optimized:
        print("Using memory-optimized configuration")
        config = create_memory_optimized_config()
    else:
        config = create_config(args.config)

    # Override config with command line arguments
    if args.data_root:
        config.data.data_root = args.data_root
    if args.exp_name:
        config.exp_name = args.exp_name

    # Setup logging
    logger = setup_logging(config.log_dir)
    logger.info(f"Starting Event Camera Semantic Segmentation - Mode: {args.mode}")
    logger.info(f"Configuration: {config.exp_name}")
    logger.info(f"Data root: {config.data.data_root}")

    # Check environment
    check_environment()

    try:
        if args.mode == 'cache':
            # Precompute cache only
            precompute_cache(config)

        elif args.mode == 'train':
            # Training (with optional resume)
            if not os.path.exists(os.path.join(config.data.data_root, 'train')):
                logger.error(f"Training data not found in {config.data.data_root}/train")
                return

            # Check if cache exists, if not, precompute it
            cache_dir = os.path.join(config.data.data_root, f'{config.data.cache_dir}_train')
            if not os.path.exists(cache_dir) or len(os.listdir(cache_dir)) == 0:
                logger.info("Cache not found, precomputing...")
                precompute_cache(config)

            train_model(config, args.checkpoint)

        elif args.mode == 'test':
            # Testing
            if not args.checkpoint:
                logger.error("Checkpoint path required for testing")
                return

            if not os.path.exists(os.path.join(config.data.data_root, 'test')):
                logger.error(f"Test data not found in {config.data.data_root}/test")
                return

            test_model(config, args.checkpoint)

        elif args.mode == 'all':
            # Complete pipeline: cache -> train -> test
            logger.info("Running complete pipeline...")

            # 1. Precompute cache
            precompute_cache(config)

            # 2. Train model
            train_model(config, args.checkpoint)

            # 3. Test with best model
            best_checkpoint = os.path.join(config.train.model_save_root, 'best_iou.pth')
            if os.path.exists(best_checkpoint):
                logger.info("Testing with best IoU model...")
                test_model(config, best_checkpoint)
            else:
                logger.warning("Best IoU checkpoint not found, skipping testing")

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Cleanup
        clear_memory()
        logger.info("Process completed")


if __name__ == '__main__':
    main()