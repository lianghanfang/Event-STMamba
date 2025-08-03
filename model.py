"""
Unified Event Camera Semantic Segmentation Model with Mamba Integration
Cleaned up version with only necessary components
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv.pytorch as spconv
from spconv.pytorch import SparseModule
import math
from typing import Optional, Dict, Any
from einops import rearrange


# ===================== Mamba Core Components =====================

class GuidanceComputer:
    """Compute guidance directions for 3D sparse data"""
    
    @staticmethod
    def spatial_centroid_shift(sparse_tensor, radius=1.0):
        """Compute spatial guidance based on neighborhood centroid"""
        indices = sparse_tensor.indices.float()
        batch_size = sparse_tensor.batch_size
        guidance_list = []
        
        for b in range(batch_size):
            batch_mask = indices[:, 0] == b
            batch_indices = indices[batch_mask, 1:4]
            
            if batch_indices.shape[0] == 0:
                continue
            
            guidance_batch = []
            for i, voxel_coord in enumerate(batch_indices):
                distances = torch.norm(batch_indices - voxel_coord.unsqueeze(0), dim=1)
                neighbor_mask = distances <= radius
                
                if neighbor_mask.sum() > 1:
                    neighbors = batch_indices[neighbor_mask]
                    centroid = neighbors.mean(dim=0)
                    direction = F.normalize((centroid - voxel_coord).unsqueeze(0), dim=1).squeeze(0)
                else:
                    direction = torch.zeros(3, device=batch_indices.device)
                
                guidance_batch.append(direction)
            
            if guidance_batch:
                guidance_list.append(torch.stack(guidance_batch))
        
        if guidance_list:
            return torch.cat(guidance_list, dim=0)
        else:
            return torch.zeros((sparse_tensor.features.shape[0], 3), device=sparse_tensor.features.device)
    
    @staticmethod
    def temporal_flow(sparse_tensor, prev_features=None):
        """Compute temporal flow guidance"""
        features = sparse_tensor.features
        
        if prev_features is None or prev_features.shape[0] != features.shape[0]:
            return torch.zeros((features.shape[0], 3), device=features.device)
        
        flow = features - prev_features
        
        # Project to 3D guidance
        guidance_net = nn.Linear(flow.shape[1], 3).to(flow.device)
        guidance = F.normalize(guidance_net(flow), dim=1)
        
        return guidance


class LearnedGuidanceNet(nn.Module):
    """Learned guidance network"""
    
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 3),
            nn.Tanh()
        )
    
    def forward(self, features):
        return F.normalize(self.net(features), dim=1)


class Mamba3DCore(nn.Module):
    """3D Mamba core with selective scan"""
    
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        
        # Mamba projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, 
                               groups=self.d_inner, padding=d_conv - 1)
        
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # Guidance integration
        self.guidance_gate = nn.Sequential(
            nn.Linear(3, self.d_inner),
            nn.LayerNorm(self.d_inner),
            nn.GELU(),
            nn.Linear(self.d_inner, self.d_inner * 3),
        )
        
        # State matrices
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
    
    def forward(self, x, guidance, indices=None):
        """Forward pass with guidance"""
        batch_size = x.shape[0]
        
        # Input projection
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        # Convolution
        x_in = rearrange(x_in, 'n d -> 1 d n')
        x_in = self.conv1d(x_in)[..., :batch_size]
        x_in = rearrange(x_in, '1 d n -> n d')
        x_in = F.silu(x_in)
        
        # Guidance gating
        gates = self.guidance_gate(guidance)
        g_input, g_state, g_output = gates.chunk(3, dim=-1)
        g_input = torch.sigmoid(g_input)
        g_state = torch.sigmoid(g_state)
        g_output = torch.sigmoid(g_output)
        
        x_modulated = x_in * g_input
        
        # SSM parameters
        A = -torch.exp(self.A_log.float())
        x_dbl = self.x_proj(x_modulated)
        B, C = x_dbl[:, :self.d_state], x_dbl[:, self.d_state:]
        
        delta = F.softplus(self.dt_proj(x_modulated))
        
        # Selective scan
        y = self.selective_scan(x_modulated, delta, A, B, C, self.D)
        
        # Output
        y = y * F.silu(z) * g_output
        return self.out_proj(y)
    
    def selective_scan(self, u, delta, A, B, C, D):
        """Simplified selective scan"""
        batch_size, d_inner = u.shape
        d_state = A.shape[1]
        
        h = torch.zeros(d_inner, d_state, device=u.device)
        outputs = []
        
        for i in range(batch_size):
            deltaA = torch.exp(delta[i].unsqueeze(-1) * A)
            deltaB = delta[i].unsqueeze(-1) * B[i].unsqueeze(0)
            
            h = deltaA * h + deltaB * u[i].unsqueeze(-1)
            y = torch.sum(h * C[i].unsqueeze(0), dim=-1) + D * u[i]
            outputs.append(y)
        
        return torch.stack(outputs, dim=0)


class Guided3DMamba(SparseModule):
    """Guided 3D Mamba module for sparse tensors"""
    
    def __init__(self, d_model: int, guidance_type: str = 'spatial',
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 radius: float = 1.0):
        super().__init__()
        
        self.d_model = d_model
        self.guidance_type = guidance_type
        self.radius = radius
        
        self.mamba_core = Mamba3DCore(d_model, d_state, d_conv, expand)
        
        if guidance_type == 'learned':
            self.learned_guidance = LearnedGuidanceNet(d_model)
        
        self.register_buffer('prev_features', None)
        self.residual_weight = nn.Parameter(torch.ones(1))
    
    def forward(self, sparse_tensor):
        features = sparse_tensor.features
        indices = sparse_tensor.indices
        
        # Compute guidance
        if self.guidance_type == 'spatial':
            guidance = GuidanceComputer.spatial_centroid_shift(sparse_tensor, self.radius)
        elif self.guidance_type == 'temporal':
            guidance = GuidanceComputer.temporal_flow(sparse_tensor, self.prev_features)
            self.prev_features = features.detach().clone()
        elif self.guidance_type == 'learned':
            guidance = self.learned_guidance(features)
        else:
            guidance = torch.zeros((features.shape[0], 3), device=features.device)
        
        # Apply Mamba
        mamba_output = self.mamba_core(features, guidance, indices)
        
        # Residual connection
        output_features = features + self.residual_weight * mamba_output
        
        return sparse_tensor.replace_feature(output_features)


# ===================== Network Building Blocks =====================

class SparseBasicBlock(spconv.SparseModule):
    """Basic residual block"""
    
    def __init__(self, inplanes, planes, stride=1, indice_key=None, norm_fn=None):
        super().__init__()
        
        self.conv1 = spconv.SubMConv3d(inplanes, planes, kernel_size=3, stride=stride,
                                      padding=1, bias=False, indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(planes, planes, kernel_size=3, stride=stride,
                                      padding=1, bias=False, indice_key=indice_key)
        self.bn2 = norm_fn(planes)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))
        
        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))
        
        return out


class MambaEnhancedBlock(spconv.SparseModule):
    """Enhanced block with Mamba"""
    
    def __init__(self, in_channels, out_channels, norm_fn, mamba_config=None):
        super().__init__()
        
        if mamba_config is None:
            mamba_config = {'guidance_type': 'spatial'}
        
        # Conv path
        self.conv_path = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, padding=1, bias=False),
            norm_fn(out_channels),
            nn.ReLU()
        )
        
        # Mamba path
        self.mamba_path = Guided3DMamba(
            d_model=out_channels,
            guidance_type=mamba_config.get('guidance_type', 'spatial'),
            d_state=mamba_config.get('d_state', 16),
            d_conv=mamba_config.get('d_conv', 4),
            expand=mamba_config.get('expand', 2),
            radius=mamba_config.get('radius', 1.0)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU()
        )
        
        # Channel matching
        if in_channels != out_channels:
            self.channel_match = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, 1, bias=False),
                norm_fn(out_channels)
            )
        else:
            self.channel_match = None
    
    def forward(self, x):
        # Conv path
        conv_out = self.conv_path(x)
        
        # Mamba path
        if self.channel_match is not None:
            mamba_input = self.channel_match(x)
        else:
            mamba_input = x
        
        mamba_out = self.mamba_path(mamba_input)
        
        # Fusion
        fused = torch.cat([conv_out.features, mamba_out.features], dim=-1)
        fused = self.fusion(fused)
        
        return conv_out.replace_feature(fused)


class PatchAttentionWithMamba(spconv.SparseModule):
    """Patch attention with optional Mamba"""
    
    def __init__(self, channel, in_spatial_size, att_spatial_size,
                 indice_key='pa1', use_mamba=True, mamba_config=None):
        super().__init__()
        
        if mamba_config is None:
            mamba_config = {'guidance_type': 'temporal'}
        
        # Downsampling
        maxpool_num = int(math.log2(in_spatial_size[0] / att_spatial_size[0]))
        self.layers = nn.ModuleList()
        
        stride = [2, 2, max(2, in_spatial_size[2] // att_spatial_size[2] // (2 ** (maxpool_num - 1)))]
        
        for i in range(maxpool_num):
            self.layers.append(spconv.SparseMaxPool3d(
                stride, stride=stride, indice_key=indice_key + str(i)
            ))
        
        # Attention
        self.multihead_attn = nn.MultiheadAttention(embed_dim=channel, num_heads=1)
        
        # Optional Mamba
        self.use_mamba = use_mamba
        if use_mamba:
            self.mamba = Guided3DMamba(
                d_model=channel,
                guidance_type=mamba_config.get('guidance_type', 'temporal'),
                d_state=mamba_config.get('d_state', 16),
                d_conv=mamba_config.get('d_conv', 4),
                expand=mamba_config.get('expand', 2),
                radius=mamba_config.get('radius', 1.0)
            )
        
        # Upsampling
        self.inver_layer = nn.ModuleList()
        for i in range(maxpool_num):
            self.inver_layer.append(spconv.SparseInverseConv3d(
                channel, channel, stride,
                indice_key=indice_key + str(maxpool_num - 1 - i),
                bias=False
            ))
        
        self.conv = spconv.SubMConv3d(channel, channel, 1, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        identity = x.features
        
        # Downsample
        for m in self.layers:
            x = m(x)
        
        # Attention
        x_features = x.features.unsqueeze(0)
        attn_output, _ = self.multihead_attn(x_features, x_features, x_features)
        attn_output = attn_output.squeeze(0)
        x = x.replace_feature(attn_output)
        
        # Optional Mamba
        if self.use_mamba:
            x = self.mamba(x)
        
        # Upsample
        for m in self.inver_layer:
            x = m(x)
        
        # Residual
        x = x.replace_feature(x.features + identity)
        x = self.conv(x)
        
        return x


# ===================== Main Network =====================

class EventSegmentationNetwork(nn.Module):
    """Event camera semantic segmentation network"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Configuration
        self.input_channels = config.get('input_channels', 4)
        self.width = config.get('width', 32)
        self.num_classes = config.get('num_classes', 1)
        
        self.mamba_config = config.get('mamba_config', {
            'use_mamba_encoder': True,
            'use_mamba_decoder': True,
            'use_mamba_attention': True,
            'encoder_config': {'guidance_type': 'spatial'},
            'decoder_config': {'guidance_type': 'learned'},
            'attention_config': {'guidance_type': 'temporal'},
        })
        
        # Norm function
        def create_norm_fn(num_channels):
            num_groups = min(8, num_channels)
            return nn.GroupNorm(num_groups, num_channels)
        
        norm_fn = create_norm_fn
        
        # Build network
        self._build_network(norm_fn)
    
    def _build_network(self, norm_fn):
        """Build the complete network"""
        # Input
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(self.input_channels, self.width, 3, padding=1,
                            bias=False, indice_key='subm1'),
            norm_fn(self.width),
            nn.ReLU()
        )
        
        # Encoder
        use_mamba = self.mamba_config.get('use_mamba_encoder', True)
        encoder_config = self.mamba_config.get('encoder_config', {})
        
        # Stage 1
        if use_mamba:
            self.conv1 = MambaEnhancedBlock(self.width, self.width, norm_fn, encoder_config)
        else:
            self.conv1 = spconv.SparseSequential(
                spconv.SubMConv3d(self.width, self.width, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(self.width),
                nn.ReLU()
            )
        
        # Stage 2
        self.conv2 = spconv.SparseSequential(
            spconv.SparseConv3d(self.width, 2 * self.width, 3, stride=[2, 2, 2],
                              padding=1, bias=False, indice_key='spconv2'),
            norm_fn(2 * self.width),
            nn.ReLU()
        )
        
        if use_mamba:
            self.conv2_2 = MambaEnhancedBlock(2 * self.width, 2 * self.width, norm_fn, encoder_config)
        else:
            self.conv2_2 = spconv.SparseSequential(
                spconv.SubMConv3d(2 * self.width, 2 * self.width, 3, padding=1, bias=False, indice_key='subm2'),
                norm_fn(2 * self.width),
                nn.ReLU()
            )
        
        self.pa2 = PatchAttentionWithMamba(
            2 * self.width, (176, 144, 128), (88, 72, 64), indice_key='pa2',
            use_mamba=self.mamba_config.get('use_mamba_attention', True),
            mamba_config=self.mamba_config.get('attention_config', {})
        )
        
        # Stage 3
        self.conv3 = spconv.SparseSequential(
            spconv.SparseConv3d(2 * self.width, 4 * self.width, 3, stride=[2, 2, 2],
                              padding=1, bias=False, indice_key='spconv3'),
            norm_fn(4 * self.width),
            nn.ReLU()
        )
        
        if use_mamba:
            self.conv3_2 = MambaEnhancedBlock(4 * self.width, 4 * self.width, norm_fn, encoder_config)
        else:
            self.conv3_2 = spconv.SparseSequential(
                spconv.SubMConv3d(4 * self.width, 4 * self.width, 3, padding=1, bias=False, indice_key='subm3'),
                norm_fn(4 * self.width),
                nn.ReLU()
            )
        
        self.pa3 = PatchAttentionWithMamba(
            4 * self.width, (88, 72, 64), (44, 36, 32), indice_key='pa3',
            use_mamba=self.mamba_config.get('use_mamba_attention', True),
            mamba_config=self.mamba_config.get('attention_config', {})
        )
        
        # Stage 4
        self.conv4 = spconv.SparseSequential(
            spconv.SparseConv3d(4 * self.width, 4 * self.width, 3, stride=[2, 2, 2],
                              padding=1, bias=False, indice_key='spconv4'),
            norm_fn(4 * self.width),
            nn.ReLU()
        )
        
        if use_mamba:
            self.conv4_2 = MambaEnhancedBlock(4 * self.width, 4 * self.width, norm_fn, encoder_config)
        else:
            self.conv4_2 = spconv.SparseSequential(
                spconv.SubMConv3d(4 * self.width, 4 * self.width, 3, padding=1, bias=False, indice_key='subm4'),
                norm_fn(4 * self.width),
                nn.ReLU()
            )
        
        self.pa4 = PatchAttentionWithMamba(
            4 * self.width, (44, 36, 32), (22, 18, 16), indice_key='pa4',
            use_mamba=self.mamba_config.get('use_mamba_attention', True),
            mamba_config=self.mamba_config.get('attention_config', {})
        )
        
        # Decoder
        use_mamba_dec = self.mamba_config.get('use_mamba_decoder', True)
        decoder_config = self.mamba_config.get('decoder_config', {})
        
        # Up 4
        self.conv_up_t4 = SparseBasicBlock(4 * self.width, 4 * self.width,
                                         indice_key='subm4', norm_fn=norm_fn)
        if use_mamba_dec:
            self.conv_up_m4 = MambaEnhancedBlock(8 * self.width, 4 * self.width, norm_fn, decoder_config)
        else:
            self.conv_up_m4 = spconv.SparseSequential(
                spconv.SubMConv3d(8 * self.width, 4 * self.width, 3, padding=1, bias=False, indice_key='subm4'),
                norm_fn(4 * self.width),
                nn.ReLU()
            )
        
        self.inv_conv4 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(4 * self.width, 4 * self.width, 3, bias=False, indice_key='spconv4'),
            norm_fn(4 * self.width),
            nn.ReLU()
        )
        
        # Up 3
        self.conv_up_t3 = SparseBasicBlock(4 * self.width, 4 * self.width,
                                         indice_key='subm3', norm_fn=norm_fn)
        if use_mamba_dec:
            self.conv_up_m3 = MambaEnhancedBlock(8 * self.width, 4 * self.width, norm_fn, decoder_config)
        else:
            self.conv_up_m3 = spconv.SparseSequential(
                spconv.SubMConv3d(8 * self.width, 4 * self.width, 3, padding=1, bias=False, indice_key='subm3'),
                norm_fn(4 * self.width),
                nn.ReLU()
            )
        
        self.inv_conv3 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(4 * self.width, 2 * self.width, 3, bias=False, indice_key='spconv3'),
            norm_fn(2 * self.width),
            nn.ReLU()
        )
        
        # Up 2
        self.conv_up_t2 = SparseBasicBlock(2 * self.width, 2 * self.width,
                                         indice_key='subm2', norm_fn=norm_fn)
        if use_mamba_dec:
            self.conv_up_m2 = MambaEnhancedBlock(4 * self.width, 2 * self.width, norm_fn, decoder_config)
        else:
            self.conv_up_m2 = spconv.SparseSequential(
                spconv.SubMConv3d(4 * self.width, 2 * self.width, 3, padding=1, bias=False, indice_key='subm2'),
                norm_fn(2 * self.width),
                nn.ReLU()
            )
        
        self.inv_conv2 = spconv.SparseSequential(
            spconv.SparseInverseConv3d(2 * self.width, self.width, 3, bias=False, indice_key='spconv2'),
            norm_fn(self.width),
            nn.ReLU()
        )
        
        # Up 1
        self.conv_up_t1 = SparseBasicBlock(self.width, self.width,
                                         indice_key='subm1', norm_fn=norm_fn)
        if use_mamba_dec:
            self.conv_up_m1 = MambaEnhancedBlock(2 * self.width, self.width, norm_fn, decoder_config)
        else:
            self.conv_up_m1 = spconv.SparseSequential(
                spconv.SubMConv3d(2 * self.width, self.width, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(self.width),
                nn.ReLU()
            )
        
        # Final
        if use_mamba_dec:
            self.conv5 = MambaEnhancedBlock(self.width, self.width, norm_fn, decoder_config)
        else:
            self.conv5 = spconv.SparseSequential(
                spconv.SubMConv3d(self.width, self.width, 3, padding=1, bias=False, indice_key='subm1'),
                norm_fn(self.width),
                nn.ReLU()
            )
        
        # Output
        self.semantic_linear = nn.Sequential(
            nn.Linear(self.width, self.num_classes),
            nn.Sigmoid()
        )
    
    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        """U-Net upsampling block"""
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = x.replace_feature(torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = x.replace_feature(x_m.features + x.features)
        x = conv_inv(x)
        return x
    
    @staticmethod
    def channel_reduction(x, out_channels):
        """Channel reduction helper"""
        features = x.features
        n, in_channels = features.shape
        x = x.replace_feature(features.view(n, out_channels, -1).sum(dim=2))
        return x
    
    def forward(self, input):
        """Forward pass"""
        # Encoder
        x = self.conv_input(input)
        x_conv1 = self.conv1(x)
        
        x_conv2 = self.conv2(x_conv1)
        x_conv2 = self.conv2_2(x_conv2)
        x_conv2 = self.pa2(x_conv2)
        
        x_conv3 = self.conv3(x_conv2)
        x_conv3 = self.conv3_2(x_conv3)
        x_conv3 = self.pa3(x_conv3)
        
        x_conv4 = self.conv4(x_conv3)
        x_conv4 = self.conv4_2(x_conv4)
        x_conv4 = self.pa4(x_conv4)
        
        # Decoder
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4,
                                    self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_conv3, x_up4, self.conv_up_t3,
                                    self.conv_up_m3, self.inv_conv3)
        x_up2 = self.UR_block_forward(x_conv2, x_up3, self.conv_up_t2,
                                    self.conv_up_m2, self.inv_conv2)
        x_up1 = self.UR_block_forward(x_conv1, x_up2, self.conv_up_t1,
                                    self.conv_up_m1, self.conv5)
        
        # Output
        output = self.semantic_linear(x_up1.features)
        
        # Ensure 1D output
        if output.dim() > 1 and output.shape[1] == 1:
            output = output.squeeze(1)
        
        # Return predictions and voxel features (for STCLoss)
        return output, x_up1


def create_model(config: Dict[str, Any]):
    """Create model from configuration"""
    return EventSegmentationNetwork(config)