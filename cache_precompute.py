"""
Standalone Cache Precomputation Script
Pre-computes voxelization cache for all datasets to reduce runtime memory usage
"""
import os
import sys
import argparse
import logging
import time
from tqdm import tqdm

from config import create_config, create_memory_optimized_config
from dataloader import EventDataset


def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def check_data_directory(data_root):
    """Check if data directories exist"""
    logger = logging.getLogger(__name__)

    if not os.path.exists(data_root):
        logger.error(f"Data root directory does not exist: {data_root}")
        return False

    modes = ['train', 'val', 'test']
    existing_modes = []

    for mode in modes:
        mode_dir = os.path.join(data_root, mode)
        if os.path.exists(mode_dir):
            files = [f for f in os.listdir(mode_dir) if f.endswith('.npz')]
            if files:
                existing_modes.append(mode)
                logger.info(f"Found {len(files)} files in {mode} directory")
            else:
                logger.warning(f"No .npz files found in {mode} directory")
        else:
            logger.warning(f"Directory does not exist: {mode_dir}")

    if not existing_modes:
        logger.error("No valid data directories found")
        return False

    return existing_modes


def precompute_cache_for_mode(config, mode):
    """Precompute cache for a specific mode (train/val/test)"""
    logger = logging.getLogger(__name__)

    logger.info(f"Starting cache precomputation for {mode} set...")
    start_time = time.time()

    try:
        # Create dataset (this will trigger cache precomputation)
        dataset = EventDataset(config, mode=mode, use_cache=True)

        # Verify cache was created
        cache_dir = os.path.join(config.data.data_root, f'{config.data.cache_dir}_{mode}')
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            logger.info(f"Cache created: {len(cache_files)} files in {cache_dir}")
        else:
            logger.warning(f"Cache directory not created: {cache_dir}")

        elapsed = time.time() - start_time
        logger.info(f"Cache precomputation for {mode} completed in {elapsed:.2f}s")

        return True

    except Exception as e:
        logger.error(f"Error precomputing cache for {mode}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Precompute voxelization cache for event camera data')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to data directory containing train/val/test folders')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--memory-optimized', action='store_true',
                        help='Use memory-optimized configuration')
    parser.add_argument('--modes', nargs='+', default=['train', 'val', 'test'],
                        choices=['train', 'val', 'test'],
                        help='Which datasets to precompute cache for')
    parser.add_argument('--max-events', type=int, default=None,
                        help='Maximum events per sample (overrides config)')
    parser.add_argument('--batch-size', type=int, default=3,
                        help='Batch size for cache computation (smaller = less memory)')

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("Starting cache precomputation...")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Modes: {args.modes}")

    # Check data directory
    existing_modes = check_data_directory(args.data_root)
    if not existing_modes:
        logger.error("No valid data found, exiting")
        return

    # Filter requested modes by existing ones
    modes_to_process = [mode for mode in args.modes if mode in existing_modes]
    if not modes_to_process:
        logger.error("None of the requested modes have valid data")
        return

    logger.info(f"Will process modes: {modes_to_process}")

    # Create configuration
    if args.memory_optimized:
        logger.info("Using memory-optimized configuration")
        config = create_memory_optimized_config()
    else:
        config = create_config(args.config)

    # Override config with command line arguments
    config.data.data_root = args.data_root

    if args.max_events:
        config.data.max_events_num = args.max_events
        logger.info(f"Override max events: {args.max_events}")

    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Max events per sample: {config.data.max_events_num}")
    logger.info(f"  Cache directory: {config.data.cache_dir}")
    logger.info(f"  Use cache: {config.data.use_cache}")

    # Process each mode
    total_start_time = time.time()
    success_count = 0

    for mode in modes_to_process:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing {mode.upper()} dataset")
        logger.info(f"{'=' * 50}")

        success = precompute_cache_for_mode(config, mode)
        if success:
            success_count += 1
        else:
            logger.error(f"Failed to precompute cache for {mode}")

    # Summary
    total_time = time.time() - total_start_time
    logger.info(f"\n{'=' * 50}")
    logger.info(f"CACHE PRECOMPUTATION SUMMARY")
    logger.info(f"{'=' * 50}")
    logger.info(f"Total time: {total_time:.2f}s")
    logger.info(f"Successful: {success_count}/{len(modes_to_process)} modes")

    if success_count == len(modes_to_process):
        logger.info("All cache precomputation completed successfully!")

        # Print cache locations
        logger.info("\nCache locations:")
        for mode in modes_to_process:
            cache_dir = os.path.join(config.data.data_root, f'{config.data.cache_dir}_{mode}')
            if os.path.exists(cache_dir):
                cache_files = len([f for f in os.listdir(cache_dir) if f.endswith('.pkl')])
                logger.info(f"  {mode}: {cache_dir} ({cache_files} files)")
    else:
        logger.error("Some cache precomputation failed!")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())