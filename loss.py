"""
Unified Loss Functions for Event Camera Semantic Segmentation
Includes STCLoss with spatial-temporal contrastive learning
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class STCLoss(nn.Module):
    """
    Spatial-Temporal Contrastive Loss for event-based segmentation
    Combines BCE loss with contrastive learning
    """
    
    def __init__(self, k=10, t=0.07, lambda_stc=0.1):
        """
        Args:
            k: Number of negative samples
            t: Temperature parameter
            lambda_stc: Weight for contrastive loss
        """
        super().__init__()
        self.k = k
        self.t = t
        self.lambda_stc = lambda_stc
        
        # Main segmentation loss
        self.bce_loss = nn.BCELoss()
        
        # Projection heads for contrastive learning
        self.projection_dim = 128
        self.spatial_proj = None
        self.temporal_proj = None
        self._proj_initialized = False
    
    def _init_projections(self, feature_dim, device):
        """Initialize projection heads based on feature dimension"""
        hidden_dim = min(feature_dim, 128)
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.projection_dim)
        ).to(device)
        
        self.temporal_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.projection_dim)
        ).to(device)
        
        self._proj_initialized = True
        self._feature_dim = feature_dim
    
    def forward(self, voxel, p2v_map, preds, labels):
        """
        Forward pass
        
        Args:
            voxel: Voxel features or SparseConvTensor
            p2v_map: Point-to-voxel mapping [N]
            preds: Model predictions [M] (voxel-level)
            labels: Ground truth labels [N] (point-level)
        
        Returns:
            total_loss: Combined loss
        """
        # Ensure correct dimensions
        if preds.dim() > 1:
            preds = preds.squeeze()
        if labels.dim() > 1:
            labels = labels.squeeze()
        
        # Main segmentation loss
        seg_loss = self._compute_segmentation_loss(preds, labels, p2v_map)
        
        # Contrastive loss (if features available)
        if hasattr(voxel, 'features') and voxel.features is not None:
            contrastive_loss = self._compute_contrastive_loss(voxel, labels, p2v_map)
            total_loss = seg_loss + self.lambda_stc * contrastive_loss
        else:
            total_loss = seg_loss
        
        return total_loss
    
    def _compute_segmentation_loss(self, preds, labels, p2v_map):
        """Compute main segmentation loss"""
        try:
            # Map voxel predictions to points
            if p2v_map is not None and len(p2v_map) > 0:
                # Ensure valid mapping
                valid_mask = (p2v_map >= 0) & (p2v_map < len(preds))
                
                if valid_mask.sum() > 0:
                    valid_p2v = p2v_map[valid_mask]
                    valid_labels = labels[valid_mask]
                    point_preds = preds[valid_p2v.long()]
                    return self.bce_loss(point_preds, valid_labels.float())
            
            # Fallback: direct comparison
            min_len = min(len(preds), len(labels))
            return self.bce_loss(preds[:min_len], labels[:min_len].float())
        
        except Exception as e:
            # Emergency fallback
            min_len = min(len(preds), len(labels))
            return self.bce_loss(preds[:min_len], labels[:min_len].float())
    
    def _compute_contrastive_loss(self, voxel, labels, p2v_map):
        """Compute spatial-temporal contrastive loss"""
        try:
            features = voxel.features
            device = features.device
            
            # Initialize projections if needed
            if not self._proj_initialized or features.shape[1] != getattr(self, '_feature_dim', -1):
                self._init_projections(features.shape[1], device)
            
            # Map labels to voxels
            voxel_labels = self._map_labels_to_voxels(labels, p2v_map, len(features))
            
            # Spatial contrastive loss
            spatial_loss = self._spatial_contrastive_loss(features, voxel_labels)
            
            # Temporal contrastive loss (if coordinates available)
            temporal_loss = torch.tensor(0.0, device=device)
            if hasattr(voxel, 'indices'):
                temporal_loss = self._temporal_contrastive_loss(features, voxel.indices, voxel_labels)
            
            return spatial_loss + temporal_loss
        
        except Exception as e:
            return torch.tensor(0.0, device=features.device if hasattr(voxel, 'features') else 'cuda')
    
    def _spatial_contrastive_loss(self, features, voxel_labels):
        """Compute spatial contrastive loss"""
        # Project features
        spatial_embeds = F.normalize(self.spatial_proj(features), dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(spatial_embeds, spatial_embeds.T) / self.t
        
        # Create masks
        positive_mask = (voxel_labels.unsqueeze(0) == voxel_labels.unsqueeze(1)) & (voxel_labels.unsqueeze(0) == 1)
        negative_mask = (voxel_labels.unsqueeze(0) != voxel_labels.unsqueeze(1))
        
        # Remove diagonal
        eye_mask = torch.eye(len(features), device=features.device).bool()
        positive_mask = positive_mask & ~eye_mask
        negative_mask = negative_mask & ~eye_mask
        
        # Compute loss
        if positive_mask.sum() > 0 and negative_mask.sum() > 0:
            pos_sim = sim_matrix[positive_mask]
            neg_sim = sim_matrix[negative_mask]
            
            # InfoNCE loss
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)
            
            loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))
            return loss
        
        return torch.tensor(0.0, device=features.device)
    
    def _temporal_contrastive_loss(self, features, indices, voxel_labels):
        """Compute temporal contrastive loss"""
        # Extract temporal coordinates
        time_coords = indices[:, 3].float()
        
        # Project features
        temporal_embeds = F.normalize(self.temporal_proj(features), dim=1)
        
        # Compute temporal similarity
        time_diff = torch.abs(time_coords.unsqueeze(0) - time_coords.unsqueeze(1))
        time_threshold = time_coords.std() * 0.5
        temporal_neighbors = time_diff < time_threshold
        
        # Remove diagonal
        eye_mask = torch.eye(len(features), device=features.device).bool()
        temporal_neighbors = temporal_neighbors & ~eye_mask
        
        # Compute similarity
        sim_matrix = torch.matmul(temporal_embeds, temporal_embeds.T) / self.t
        
        if temporal_neighbors.sum() > 0:
            pos_sim = sim_matrix[temporal_neighbors]
            neg_sim = sim_matrix[~temporal_neighbors & ~eye_mask]
            
            if len(neg_sim) > 0:
                pos_exp = torch.exp(pos_sim)
                neg_exp = torch.exp(neg_sim)
                
                loss = -torch.log(pos_exp.sum() / (pos_exp.sum() + neg_exp.sum() + 1e-8))
                return loss
        
        return torch.tensor(0.0, device=features.device)
    
    def _map_labels_to_voxels(self, point_labels, p2v_map, num_voxels):
        """Map point-level labels to voxel-level"""
        voxel_labels = torch.zeros(num_voxels, device=point_labels.device)
        
        if p2v_map is not None and len(p2v_map) > 0:
            # Valid mapping check
            valid_mask = (p2v_map >= 0) & (p2v_map < num_voxels)
            
            if valid_mask.sum() > 0:
                valid_p2v = p2v_map[valid_mask]
                valid_labels = point_labels[valid_mask]
                
                # Aggregate labels (majority voting)
                voxel_label_sum = torch.zeros(num_voxels, device=point_labels.device)
                voxel_count = torch.zeros(num_voxels, device=point_labels.device)
                
                voxel_label_sum.scatter_add_(0, valid_p2v.long(), valid_labels.float())
                voxel_count.scatter_add_(0, valid_p2v.long(), torch.ones_like(valid_labels, dtype=torch.float))
                
                # Avoid division by zero
                voxel_count = voxel_count.clamp(min=1)
                voxel_labels = (voxel_label_sum / voxel_count > 0.5).float()
        
        return voxel_labels


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, preds, targets):
        """
        Args:
            preds: Predictions (after sigmoid)
            targets: Ground truth binary labels
        """
        # Ensure correct dimensions
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Focal loss computation
        bce_loss = F.binary_cross_entropy(preds, targets, reduction='none')
        
        # Apply focal term
        pt = torch.where(targets == 1, preds, 1 - preds)
        focal_term = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * focal_term * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined loss with multiple components"""
    
    def __init__(self, loss_weights=None):
        super().__init__()
        
        if loss_weights is None:
            loss_weights = {
                'bce': 1.0,
                'focal': 0.5,
                'stc': 0.1
            }
        
        self.loss_weights = loss_weights
        
        # Individual losses
        self.bce = nn.BCELoss()
        self.focal = FocalLoss()
        self.stc = STCLoss()
    
    def forward(self, voxel, p2v_map, preds, labels):
        """Compute combined loss"""
        total_loss = 0.0
        
        # Map predictions to points
        if p2v_map is not None and len(p2v_map) > 0:
            valid_mask = (p2v_map >= 0) & (p2v_map < len(preds))
            if valid_mask.sum() > 0:
                point_preds = preds[p2v_map[valid_mask].long()]
                point_labels = labels[valid_mask]
            else:
                min_len = min(len(preds), len(labels))
                point_preds = preds[:min_len]
                point_labels = labels[:min_len]
        else:
            min_len = min(len(preds), len(labels))
            point_preds = preds[:min_len]
            point_labels = labels[:min_len]
        
        # BCE Loss
        if 'bce' in self.loss_weights:
            total_loss += self.loss_weights['bce'] * self.bce(point_preds, point_labels.float())
        
        # Focal Loss
        if 'focal' in self.loss_weights:
            total_loss += self.loss_weights['focal'] * self.focal(point_preds, point_labels.float())
        
        # STC Loss
        if 'stc' in self.loss_weights:
            total_loss += self.loss_weights['stc'] * self.stc(voxel, p2v_map, preds, labels)
        
        return total_loss