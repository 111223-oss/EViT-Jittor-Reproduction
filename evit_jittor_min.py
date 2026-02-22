# coding: utf-8
import math
import numpy as np
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1  # 有GPU就用GPU，没有会自动退回CPU


def layer_norm(x, eps=1e-5):
    # x: [B, N, C]
    mean = x.mean(dim=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdims=True)
    return (x - mean) / jt.sqrt(var + eps)


class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_chans=3, embed_dim=192):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv(in_chans, embed_dim, kernel_size=patch, stride=patch)

    def execute(self, x):
        # x: [B,3,32,32] -> [B, C, H/ps, W/ps] -> [B, N, C]
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).transpose(0, 2, 1)  # [B, N, C]
        return x


class MSA(nn.Module):
    def __init__(self, dim=192, heads=3):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.d = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def execute(self, x):
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.qkv(x)  # [B, N, 3C]
        qkv = qkv.reshape(B, N, 3, self.heads, self.d).transpose(0, 3, 2, 1, 4)  # [B,h,3,N,d]
        q = qkv[:, :, 0]  # [B,h,N,d]
        k = qkv[:, :, 1]
        v = qkv[:, :, 2]

        attn = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.d)  # [B,h,N,N]
        attn = nn.softmax(attn, dim=-1)
        out = attn @ v  # [B,h,N,d]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out, attn


class MLP(nn.Module):
    def __init__(self, dim=192, mlp_ratio=4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim=192, heads=3, mlp_ratio=4):
        super().__init__()
        self.msa = MSA(dim, heads)
        self.mlp = MLP(dim, mlp_ratio)

    def execute(self, x):
        # Pre-LN
        y, attn = self.msa(layer_norm(x))
        x = x + y
        x = x + self.mlp(layer_norm(x))
        return x, attn


class EViTMin(nn.Module):
    """
    简化版EViT:
    - ViT backbone
    - 用 CLS 对 patch 的注意力做 topk，生成 mask（不改 token 数量，只把不保留的 token 置0）
    这样最稳、最不容易在 Jittor 上卡在 gather/shape 上。
    """
    def __init__(self,
                 img_size=32,
                 patch=4,
                 in_chans=3,
                 num_classes=10,
                 embed_dim=192,
                 depth=6,
                 heads=3,
                 mlp_ratio=4,
                 keep_rate=0.5,
                 prune_after=2):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch, in_chans, embed_dim)

        # token 数： (32/4)^2 = 64，再加 CLS -> 65
        self.num_patches = (img_size // patch) * (img_size // patch)

        self.cls = jt.zeros((1, 1, embed_dim))
        self.pos = jt.randn((1, 1 + self.num_patches, embed_dim)) * 0.02

        self.blocks = nn.ModuleList([Block(embed_dim, heads, mlp_ratio) for _ in range(depth)])
        self.head = nn.Linear(embed_dim, num_classes)

        self.keep_rate = keep_rate
        self.prune_after = prune_after  # 从第几个 block 开始做mask

    def execute(self, x):
        # x: [B,3,32,32]
        B = x.shape[0]
        tok = self.patch_embed(x)  # [B,64,C]

        cls = self.cls.repeat(B, 1, 1)  # [B,1,C]
        x = jt.concat([cls, tok], dim=1)  # [B,65,C]
        x = x + self.pos

        # token mask: [B, N], CLS always 1
        mask = jt.ones((B, 1 + self.num_patches))

        left_tokens = max(1, int(self.num_patches * self.keep_rate))  # patch里保留多少个

        for i, blk in enumerate(self.blocks):
            x, attn = blk(x)

            if i >= self.prune_after:
                # 用 numpy 做 topk（稳定、不会掉进 jt.topk/tuple 的坑）
                # attn: [B,h,N,N] 取 CLS->patch 的注意力
                a = attn.data  # numpy
                cls_attn = a[:, :, 0, 1:]           # [B,h,64]
                cls_attn = cls_attn.mean(axis=1)    # [B,64]

                idx = np.argsort(-cls_attn, axis=1)[:, :left_tokens]  # [B,left]
                m = np.zeros((B, 1 + self.num_patches), dtype=np.float32)
                m[:, 0] = 1.0
                for b in range(B):
                    m[b, 1 + idx[b]] = 1.0

                mask = jt.array(m)

                # 只做mask，不改 token 数
                x = x * mask.unsqueeze(-1)

        cls_out = x[:, 0]  # CLS
        logits = self.head(cls_out)
        return logits
