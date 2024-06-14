import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from segment_anything.modeling import window_partition, window_unpartition

class Sam3DWrapper(nn.Module):
    def _get_adapter_3d():
        pass

    def __init__(
        self,
        sam: nn.Module,
        **kwargs
    ):
        super().__init__()
        self.sam = sam
        self.image_encoder = self.sam.model.image_encoder
        self.img_size = self.image_encoder.img_size
        
        # create adapter blocks
        n_blocks = len(self.image_encoder.blocks)
        
        # freeze sam blocks
        for k, v in self.image_encoder.named_parameters():
            if '.adapter_' not in k:
                v.requires_grad = False
        
        # adapter blocks before self attention
        adapter_list1 = []
        # adapter blocks after self attention
        adapter_list2 = []
        for i in range(n_blocks):
            adapter_list1.append(self._get_adapter_3d())
            adapter_list2.append(self._get_adapter_3d())
        
        self.adapter_list1 = nn.ModuleList(adapter_list1)
        self.adapter_list2 = nn.ModuleList(adapter_list2)
        
    def forward(self, x: torch.Tensor, d_size) -> torch.Tensor:
        x = self.image_encoder.patch_embed(x)
        if self.image_encoder.pos_embed is not None:
            x = x + self.image_encoder.pos_embed

        for blk in self.image_encoder.blocks:
            x = blk(x, d_size)

        x = self.image_encoder.neck(x.permute(0, 3, 1, 2))

        return x


class ND_Block(nn.Module):
    def __init__(
        self,
        Block: nn.Module
    ) -> None:
        super.__init__()
        self.Block = Block

    def forward(self, x: torch.Tensor, d_size) -> torch.Tensor:
        b_size, hw_size = x.shape[0], x.shape[1]
        
        # 3D adapter
        x = self.Block.adapter_norm(x)
        x = self.Block.adapter_linear_down(x)
        x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.Block.adapter_conv(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        x = self.Block.adapter_act(x)
        x = self.Block.adapter_linear_up(x)
        x = shortcut + x
        # end 3D adapter
        
        shortcut = x
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        
        # 3D adapter
        shortcut = x
        x = self.Block.adapter_norm_2(x)
        x = self.Block.adapter_linear_down_2(x)
        x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        x = torch.permute(x, (0, -1, 1, 2, 3))
        x = self.Block.adapter_conv_2(x)
        x = torch.permute(x, (0, 2, 3, 4, 1))
        x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        x = self.Block.adapter_act_2(x)
        x = self.Block.adapter_linear_up_2(x)
        x = shortcut + x
        # end 3D adapter
        
        x = x + self.mlp(self.norm2(x))

        return x
