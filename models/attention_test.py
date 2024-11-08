import torch
import torch.nn as nn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 初始化参数
point_num = 512
batch_size = 18
image_dim = 64  # query 的维度
point_dim = 256        # key 的维度
# num_heads = 8    # 多头数量
dropout = 0.1    # Dropout概率

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 示例输入数据
# query = torch.randn(point_num, batch_size, embed_dim)  # [sequence length, batch size, feature dimension]
# key = torch.randn(point_num, batch_size, kdim)
# value = torch.randn(point_num, batch_size, vdim)

from flash_attn.modules.mha import FlashCrossAttention
from einops import rearrange

class CrossModalAttention(nn.Module):
    def __init__(self, point_dim, img_dim, dropout, normalize=False, in_proj_bias=True, out_proj_bias=True):
        super(CrossModalAttention, self).__init__()
        self.feat_dim = self.point_dim = point_dim
        self.img_dim = img_dim
        self.normalize = normalize
        self.dropout = dropout
        
        if self.normalize:
            self.point_norm = nn.LayerNorm(self.point_dim)
            self.img_norm = nn.LayerNorm(self.img_dim)

        self.img_mlp = nn.Sequential(
            nn.Conv1d(img_dim, 128, 1),
            nn.BatchNorm1d(128), 
            nn.ReLU(inplace=True),
            nn.Conv1d(128, self.feat_dim, 1)
        )
        
        self.point_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        self.point_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
        self.img_q_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=in_proj_bias)
        self.img_kv_proj = nn.Linear(self.feat_dim, 2 * self.feat_dim, bias=in_proj_bias)
        
        self.point_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        self.image_cross_attn = FlashCrossAttention(attention_dropout=dropout)
        # self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        # self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
    def forward(self, point_feat, img_feat):
        if self.normalize:
            point_feat = self.point_norm(point_feat)
            img_feat = self.img_norm(img_feat)

        Bs, N, _ = point_feat.shape
        # 调整维度符合多头注意力输入要求 (Seq_len, Batch, Embedding_dim)
        # point_feat = point_feat.transpose(0, 1)  # (num_pc, B, point_dim)
        # img_feat = img_feat.transpose(0, 1)  # (num_pc, B, img_dim)
        
        img_feat = self.img_mlp(img_feat.permute(0, 2, 1)).permute(0, 2, 1)
        
        img_q = self.img_q_proj(img_feat).half()
        img_kv = self.img_kv_proj(img_feat).half()
        point_q = self.point_q_proj(point_feat).half()
        point_kv = self.point_kv_proj(point_feat).half()
        
        img_q = rearrange(img_q, "... (h d) -> ... h d", d=self.feat_dim)
        point_kv = rearrange(point_kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
        point_fuse = self.point_cross_attn(img_q, point_kv)
        point_fuse = point_feat + point_fuse.view(Bs, N, -1).float()
        
        point_q = rearrange(point_q, "... (h d) -> ... h d", d=self.feat_dim)
        img_kv = rearrange(img_kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.feat_dim)
        image_fuse = self.image_cross_attn(point_q, img_kv)
        image_fuse = img_feat + image_fuse.view(Bs, N, -1).float()
        
        fused_feat = torch.concat([point_fuse, image_fuse], dim=2)
        fused_feat = fused_feat.permute((0, 2, 1))  # (B, output_dim, num_pc)
        return fused_feat
    

model = CrossModalAttention(point_dim=point_dim, img_dim=image_dim, dropout=dropout, normalize=False).to(device)

# 定义示例输入
point_feat = torch.randn(batch_size, point_num, point_dim).to(device)  # 点云特征 (B, N_points, point_dim)
img_feat = torch.randn(batch_size, point_num, image_dim).to(device)        # 图像特征 (B, N_points, img_dim)

# 运行前向传播
output = model(point_feat, img_feat)

# 检查输出形状
print("Output shape:", output.shape)