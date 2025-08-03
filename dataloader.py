"""
Memory-Efficient Event Camera Dataset with Precomputed Voxelization Cache
Integrated optimizations to reduce CPU memory usage during training
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import spconv.pytorch as spconv
import logging
import pickle
import gc
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


def optimized_voxelization_idx(coords, batchsize, mode=4):
    """Memory-efficient voxelization with reduced memory footprint"""
    coords = coords.long()
    N = coords.shape[0]

    # Create unique coordinate hash with smaller multipliers to avoid overflow
    coord_hash = (coords[:, 0] * 100000000 +
                  coords[:, 1] * 100000 +
                  coords[:, 2] * 1000 +
                  coords[:, 3])

    # Find unique coordinates
    unique_hash, inverse_indices = torch.unique(coord_hash, return_inverse=True)
    unique_coords = torch.zeros((len(unique_hash), 4), dtype=torch.long, device=coords.device)

    # Reconstruct unique coordinates efficiently
    for i, hash_val in enumerate(unique_hash):
        mask = coord_hash == hash_val
        idx = mask.nonzero(as_tuple=True)[0][0]
        unique_coords[i] = coords[idx]

    # Create output_map with reduced memory footprint
    max_points = 16  # Further reduced from 32
    output_map = torch.zeros((len(unique_hash), max_points + 1), dtype=torch.int32, device=coords.device)

    for i in range(len(unique_hash)):
        mask = inverse_indices == i
        indices = mask.nonzero(as_tuple=True)[0]
        count = min(len(indices), max_points)
        output_map[i, 0] = count
        if count > 0:
            output_map[i, 1:count+1] = indices[:count].int()

    return unique_coords, inverse_indices.int(), output_map


def optimized_voxelization(feats, map_rule, mode=4):
    """Memory-efficient feature aggregation"""
    M, max_points_plus_one = map_rule.shape
    max_points = max_points_plus_one - 1
    C = feats.shape[1]

    output_feats = torch.zeros((M, C), dtype=feats.dtype, device=feats.device)

    for i in range(M):
        count = map_rule[i, 0].item()
        if count > 0:
            indices = map_rule[i, 1:count+1]
            valid_indices = indices[indices < feats.shape[0]]
            if len(valid_indices) > 0:
                if mode == 4:  # mean
                    output_feats[i] = feats[valid_indices].mean(dim=0)
                else:  # sum
                    output_feats[i] = feats[valid_indices].sum(dim=0)

    return output_feats


class EventDataset(Dataset):
    """Memory-efficient Event Camera Dataset with aggressive caching"""

    def __init__(self, config, mode='train', use_cache=True):
        """
        Args:
            config: Configuration object
            mode: 'train', 'val', or 'test'
            use_cache: Whether to use precomputed voxelization cache
        """
        # Setup logging first before any operations
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(f"{__name__}.{mode}")

        self.config = config
        self.mode = mode
        self.root = os.path.join(config.data.data_root, mode)
        self.use_cache = use_cache and config.data.use_cache

        # Get file list
        self.file_list = self._get_file_list()

        # Setup cache directory
        self.cache_dir = os.path.join(config.data.data_root, f'{config.data.cache_dir}_{mode}')
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._precompute_cache()

    def _get_file_list(self) -> List[str]:
        """Get list of data files"""
        if not os.path.exists(self.root):
            raise ValueError(f"Data directory {self.root} does not exist")

        file_list = sorted([f for f in os.listdir(self.root) if f.endswith('.npz')])

        if len(file_list) == 0:
            raise ValueError(f"No .npz files found in {self.root}")

        # Use print instead of logger since this might be called before logger init
        print(f"Found {len(file_list)} files in {self.root}")
        return file_list

    def _precompute_cache(self):
        """Precompute voxelization cache for ALL files to avoid runtime computation"""
        self.logger.info(f"Checking cache for {self.mode} set...")

        missing_files = []
        for idx in range(len(self.file_list)):
            cache_file = os.path.join(self.cache_dir, f'{idx}_voxel.pkl')
            if not os.path.exists(cache_file):
                missing_files.append(idx)

        if not missing_files:
            self.logger.info(f"Cache complete for {self.mode} set ({len(self.file_list)} files)")
            return

        self.logger.info(f"Computing cache for {len(missing_files)} missing files...")

        # Process in small batches to avoid memory overflow
        batch_size = 3  # Very small batch to minimize memory usage
        for i in tqdm(range(0, len(missing_files), batch_size), desc=f"Caching {self.mode}"):
            batch_indices = missing_files[i:i+batch_size]

            for idx in batch_indices:
                try:
                    self._cache_single_sample(idx)
                except Exception as e:
                    self.logger.warning(f"Failed to cache sample {idx}: {e}")
                    continue

            # Aggressive memory cleanup after each batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.logger.info(f"Cache computation completed for {self.mode} set")

    def _cache_single_sample(self, idx: int):
        """Cache voxelization for a single sample with memory optimization"""
        cache_file = os.path.join(self.cache_dir, f'{idx}_voxel.pkl')
        if os.path.exists(cache_file):
            return

        try:
            # Load raw data
            file_path = os.path.join(self.root, self.file_list[idx])
            events = np.load(file_path)

            # Extract data (ensure minimal memory footprint)
            evs_norm = events['evs_norm'].astype(np.float32)
            ev_loc = events['ev_loc'].astype(np.float32)
            seg_label = evs_norm[:, 4]
            idx_label = evs_norm[:, 5]
            evs_norm = evs_norm[:, 0:4]

            # Aggressive downsampling for memory efficiency
            max_events = min(self.config.data.max_events_num, 30000)  # Further limit
            if len(ev_loc) > max_events:
                indices = np.random.choice(len(ev_loc), max_events, replace=False)
                ev_loc = ev_loc[indices]
                evs_norm = evs_norm[indices]
                seg_label = seg_label[indices]
                idx_label = idx_label[indices]

            # Add batch index and ensure contiguous memory
            locs = np.hstack((np.zeros((ev_loc.shape[0], 1)), ev_loc)).astype(np.int32)

            # Convert to tensors on CPU (avoid GPU memory during caching)
            locs_tensor = torch.from_numpy(locs).long().contiguous()
            features_tensor = torch.from_numpy(evs_norm).float().contiguous()

            # Voxelization on CPU
            voxel_locs, p2v_map, v2p_map = optimized_voxelization_idx(locs_tensor, 1, 4)
            voxel_feats = optimized_voxelization(features_tensor, v2p_map, 4)

            # Save to cache with compression
            cache_data = {
                'voxel_feats': voxel_feats.cpu().numpy().astype(np.float32),
                'voxel_coords': voxel_locs.cpu().numpy().astype(np.int32),
                'p2v_map': p2v_map.cpu().numpy().astype(np.int32),
                'seg_label': seg_label.astype(np.float32),
                'idx_label': idx_label.astype(np.float32),
                'spatial_shape': np.array([352, 288, 256], dtype=np.int32)  # Fixed shape
            }

            # Save with compression
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Clear intermediate tensors
            del locs_tensor, features_tensor, voxel_locs, voxel_feats, p2v_map, v2p_map

        except Exception as e:
            self.logger.warning(f"Error caching sample {idx}: {e}")
        finally:
            # Ensure cleanup
            gc.collect()

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        """Get a single sample from cache (no runtime computation)"""
        try:
            # Always try to load from cache first
            cache_file = os.path.join(self.cache_dir, f'{idx}_voxel.pkl')

            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data
            else:
                # If cache miss, compute on-the-fly (should be rare)
                self.logger.warning(f"Cache miss for sample {idx}, computing on-the-fly")
                return self._compute_on_the_fly(idx)

        except Exception as e:
            self.logger.warning(f"Error loading sample {idx}: {e}")
            return None

    def _compute_on_the_fly(self, idx: int) -> Optional[Dict]:
        """Fallback: compute voxelization on-the-fly if cache missing"""
        try:
            file_path = os.path.join(self.root, self.file_list[idx])
            events = np.load(file_path)

            evs_norm = events['evs_norm'].astype(np.float32)
            ev_loc = events['ev_loc'].astype(np.float32)
            seg_label = evs_norm[:, 4]
            idx_label = evs_norm[:, 5]
            evs_norm = evs_norm[:, 0:4]

            # Aggressive downsampling
            max_events = min(self.config.data.max_events_num, 20000)
            if len(ev_loc) > max_events:
                indices = np.random.choice(len(ev_loc), max_events, replace=False)
                ev_loc = ev_loc[indices]
                evs_norm = evs_norm[indices]
                seg_label = seg_label[indices]
                idx_label = idx_label[indices]

            return {
                'ev_loc': ev_loc,
                'evs_norm': evs_norm,
                'seg_label': seg_label,
                'idx_label': idx_label
            }
        except Exception as e:
            self.logger.error(f"Failed to compute sample {idx}: {e}")
            return None


def collate_fn_events(batch: List[Dict]) -> Optional[Dict]:
    """Memory-efficient collate function with aggressive optimization"""
    # Filter None values
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    batch_size = len(batch)

    # Check if data is pre-voxelized (cache hit)
    if 'voxel_feats' in batch[0]:
        return collate_cached_data(batch, batch_size)
    else:
        return collate_raw_data(batch, batch_size)


def collate_cached_data(batch: List[Dict], batch_size: int) -> Optional[Dict]:
    """Collate pre-cached voxelized data"""
    try:
        all_feats = []
        all_coords = []
        all_p2v = []
        all_seg = []
        all_idx = []
        voxel_offset = 0

        for i, item in enumerate(batch):
            # Update batch index in coordinates
            coords = item['voxel_coords'].copy()
            coords[:, 0] = i

            all_feats.append(item['voxel_feats'])
            all_coords.append(coords)
            all_p2v.append(item['p2v_map'] + voxel_offset)
            all_seg.append(item['seg_label'])
            all_idx.append(item['idx_label'])

            voxel_offset += len(item['voxel_feats'])

        # Concatenate with memory optimization
        voxel_feats = torch.from_numpy(np.concatenate(all_feats, axis=0)).float()
        voxel_coords = torch.from_numpy(np.concatenate(all_coords, axis=0)).int()
        p2v_map = torch.from_numpy(np.concatenate(all_p2v, axis=0)).int()
        seg_labels = torch.from_numpy(np.concatenate(all_seg, axis=0)).float()
        idx_labels = torch.from_numpy(np.concatenate(all_idx, axis=0)).float()

        # Move to CUDA if available and configured
        if torch.cuda.is_available():
            voxel_feats = voxel_feats.cuda()
            voxel_coords = voxel_coords.cuda()

        # Use fixed spatial shape to avoid shape issues
        spatial_shape = [352, 288, 256]

        # Create sparse tensor
        voxel_ev = spconv.SparseConvTensor(
            voxel_feats,
            voxel_coords,
            spatial_shape,
            batch_size
        )

        # Create dummy locs for compatibility
        total_points = sum(len(item['seg_label']) for item in batch)
        locs = torch.zeros((total_points, 4), dtype=torch.float32)

        point_offset = 0
        for i, item in enumerate(batch):
            num_points = len(item['seg_label'])
            locs[point_offset:point_offset+num_points, 0] = i
            # Fill other dimensions with dummy values
            locs[point_offset:point_offset+num_points, 1:] = torch.randn(num_points, 3) * 0.1
            point_offset += num_points

        return {
            'voxel_ev': voxel_ev,
            'seg_label': seg_labels,
            'p2v_map': p2v_map,
            'idx_label': idx_labels,
            'locs': locs
        }

    except Exception as e:
        logging.error(f"Error in collate_cached_data: {e}")
        return None


def collate_raw_data(batch: List[Dict], batch_size: int) -> Optional[Dict]:
    """Collate raw event data with online voxelization (fallback)"""
    try:
        all_locs = []
        all_feats = []
        all_seg = []
        all_idx = []

        for i, item in enumerate(batch):
            # Add batch index
            locs = np.hstack((i * np.ones((len(item['ev_loc']), 1)), item['ev_loc']))
            all_locs.append(locs)
            all_feats.append(item['evs_norm'])
            all_seg.append(item['seg_label'])
            all_idx.append(item['idx_label'])

        # Concatenate
        locs = torch.from_numpy(np.concatenate(all_locs, axis=0)).long()
        features = torch.from_numpy(np.concatenate(all_feats, axis=0)).float()
        seg_labels = torch.from_numpy(np.concatenate(all_seg, axis=0)).float()
        idx_labels = torch.from_numpy(np.concatenate(all_idx, axis=0)).float()

        # Voxelize
        voxel_coords, p2v_map, v2p_map = optimized_voxelization_idx(locs, batch_size)

        if torch.cuda.is_available():
            features = features.cuda()
            v2p_map = v2p_map.cuda()

        voxel_feats = optimized_voxelization(features, v2p_map)

        if torch.cuda.is_available():
            voxel_feats = voxel_feats.cuda()
            voxel_coords = voxel_coords.cuda()

        # Create sparse tensor
        spatial_shape = [352, 288, 256]

        voxel_ev = spconv.SparseConvTensor(
            voxel_feats,
            voxel_coords.int(),
            spatial_shape,
            batch_size
        )

        return {
            'voxel_ev': voxel_ev,
            'seg_label': seg_labels,
            'p2v_map': p2v_map,
            'idx_label': idx_labels,
            'locs': locs.float()
        }

    except Exception as e:
        logging.error(f"Error in collate_raw_data: {e}")
        return None


def create_dataloader(config, mode='train'):
    """Create memory-optimized dataloader"""
    dataset = EventDataset(config, mode=mode, use_cache=config.data.use_cache)

    # Memory-optimized dataloader settings
    dataloader_config = {
        'batch_size': config.data.batch_size,
        'shuffle': (mode == 'train' and config.data.shuffle_train),
        'num_workers': 0,  # Force single worker to reduce memory overhead
        'collate_fn': collate_fn_events,
        'pin_memory': False,  # Disable pin_memory to save memory
        'drop_last': config.data.drop_last,
        'persistent_workers': False
    }

    dataloader = DataLoader(dataset, **dataloader_config)
    
    return dataloader