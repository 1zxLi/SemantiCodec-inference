# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from json import encoder

import torch
import torch.nn as nn

from timm.models.vision_transformer import Block
from semanticodec.modules.audiomae.pos_embed import (
    get_2d_sincos_pos_embed,
    get_2d_sincos_pos_embed_flexible,
    get_1d_sincos_pos_embed_from_grid,
)
from semanticodec.modules.audiomae.patch_embed import PatchEmbed_new, PatchEmbed_org


class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        stride=10,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        audio_exp=False,
        alpha=0.0,
        temperature=0.2,
        mode=0,
        contextual_depth=8,
        use_custom_patch=False,
        split_pos=False,
        pos_trainable=False,
        use_nce=False,
        beta=4.0,
        decoder_mode=0,
        mask_t_prob=0.6,
        mask_f_prob=0.5,
        mask_2d=False,
        epoch=0,
        no_shift=False,
    ):
        super().__init__()

        self.audio_exp = audio_exp
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        if use_custom_patch:
            print(
                f"Use custom patch_emb with patch size: {patch_size}, stride: {stride}"
            )
            self.patch_embed = PatchEmbed_new(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                stride=stride,
            )
        else:
            self.patch_embed = PatchEmbed_org(img_size, patch_size, in_chans, embed_dim)
        self.use_custom_patch = use_custom_patch
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # self.split_pos = split_pos # not useful
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=pos_trainable
        )  # fixed sin-cos embedding

        self.encoder_depth = depth
        self.contextual_depth = contextual_depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )  # qk_scale=None
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=pos_trainable,
        )  # fixed sin-cos embedding

        self.no_shift = no_shift

        self.decoder_mode = decoder_mode
        if (
            self.use_custom_patch
        ):  # overlapped patches as in AST. Similar performance yet compute heavy
            window_size = (6, 6)
            feat_size = (102, 12)
        else:
            window_size = (4, 4)
            feat_size = (64, 8)
        if self.decoder_mode == 1:
            decoder_modules = []
            for index in range(16):
                if self.no_shift:
                    shift_size = (0, 0)
                else:
                    if (index % 2) == 0:
                        shift_size = (0, 0)
                    else:
                        shift_size = (2, 0)
                    # shift_size = tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size])
                decoder_modules.append(
                    SwinTransformerBlock(
                        dim=decoder_embed_dim,
                        num_heads=16,
                        feat_size=feat_size,
                        window_size=window_size,
                        shift_size=shift_size,
                        mlp_ratio=mlp_ratio,
                        drop=0.0,
                        drop_attn=0.0,
                        drop_path=0.0,
                        extra_norm=False,
                        sequential_attn=False,
                        norm_layer=norm_layer,  # nn.LayerNorm,
                    )
                )
            self.decoder_blocks = nn.ModuleList(decoder_modules)
        else:
            # Transfomer
            self.decoder_blocks = nn.ModuleList(
                [
                    Block(
                        decoder_embed_dim,
                        decoder_num_heads,
                        mlp_ratio,
                        qkv_bias=True,
                        norm_layer=norm_layer,
                    )  # qk_scale=None,
                    for i in range(decoder_depth)
                ]
            )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.patch_size = patch_size
        self.stride = stride

        # audio exps
        self.alpha = alpha
        self.T = temperature
        self.mode = mode
        self.use_nce = use_nce
        self.beta = beta

        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.mask_t_prob = mask_t_prob
        self.mask_f_prob = mask_f_prob
        self.mask_2d = mask_2d

        self.epoch = epoch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.audio_exp:
            pos_embed = get_2d_sincos_pos_embed_flexible(
                self.pos_embed.shape[-1], self.patch_embed.patch_hw, cls_token=True
            )
        else:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.audio_exp:
            decoder_pos_embed = get_2d_sincos_pos_embed_flexible(
                self.decoder_pos_embed.shape[-1],
                self.patch_embed.patch_hw,
                cls_token=True,
            )
        else:
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1],
                int(self.patch_embed.num_patches**0.5),
                cls_token=True,
            )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        if self.audio_exp:
            if self.use_custom_patch:  # overlapped patch
                h, w = self.patch_embed.patch_hw
                # todo: fixed h/w patch size and stride size. Make hw custom in the future
                x = imgs.unfold(2, self.patch_size, self.stride).unfold(
                    3, self.patch_size, self.stride
                )  # n,1,H,W -> n,1,h,w,p,p
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
                # x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                # x = torch.einsum('nchpwq->nhwpqc', x)
                # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
            else:
                h = imgs.shape[2] // p
                w = imgs.shape[3] // p
                # h,w = self.patch_embed.patch_hw
                x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
                x = torch.einsum("nchpwq->nhwpqc", x)
                x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        else:
            h = w = imgs.shape[2] // p
            x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
            x = torch.einsum("nchpwq->nhwpqc", x)
            x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = 1024 // p
        w = 128 // p
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum("nhwpqc->nchpwq", x)
        specs = x.reshape(shape=(x.shape[0], 1, h * p, w * p))
        return specs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        """
        2D: Spectrogram (msking t and f under mask_t_prob and mask_f_prob)
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        if self.use_custom_patch:  # overlapped patch
            T = 101
            F = 12
        else:
            T = 64
            F = 8
        # x = x.reshape(N, T, F, D)
        len_keep_t = int(T * (1 - mask_t_prob))
        len_keep_f = int(F * (1 - mask_f_prob))

        # noise for mask in time
        noise_t = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample aling time
        ids_shuffle_t = torch.argsort(
            noise_t, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore_t = torch.argsort(ids_shuffle_t, dim=1)
        ids_keep_t = ids_shuffle_t[:, :len_keep_t]
        # noise mask in freq
        noise_f = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        ids_shuffle_f = torch.argsort(
            noise_f, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore_f = torch.argsort(ids_shuffle_f, dim=1)
        ids_keep_f = ids_shuffle_f[:, :len_keep_f]  #

        # generate the binary mask: 0 is keep, 1 is remove
        # mask in freq
        mask_f = torch.ones(N, F, device=x.device)
        mask_f[:, :len_keep_f] = 0
        mask_f = (
            torch.gather(mask_f, dim=1, index=ids_restore_f)
            .unsqueeze(1)
            .repeat(1, T, 1)
        )  # N,T,F
        # mask in time
        mask_t = torch.ones(N, T, device=x.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = (
            torch.gather(mask_t, dim=1, index=ids_restore_t)
            .unsqueeze(1)
            .repeat(1, F, 1)
            .permute(0, 2, 1)
        )  # N,T,F
        mask = 1 - (1 - mask_t) * (1 - mask_f)  # N, T, F

        # get masked x
        id2res = torch.Tensor(list(range(N * T * F))).reshape(N, T, F).to(x.device)
        id2res = id2res + 999 * mask  # add a large value for masked elements
        id2res2 = torch.argsort(id2res.flatten(start_dim=1))
        ids_keep = id2res2.flatten(start_dim=1)[:, : len_keep_f * len_keep_t]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        ids_restore = torch.argsort(id2res2.flatten(start_dim=1))
        mask = mask.flatten(start_dim=1)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask_2d=False):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_2d:
            x, mask, ids_restore = self.random_masking_2d(
                x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob
            )
        else:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, None

    def forward_encoder_no_random_mask_no_average(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        # if mask_2d:
        #     x, mask, ids_restore = self.random_masking_2d(x, mask_t_prob=self.mask_t_prob, mask_f_prob=self.mask_f_prob)
        # else:
        #     x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_encoder_no_mask(self, x):
        # embed patches
        # 生成上下文嵌入  无随机掩码的输入  将输入梅尔频谱图转换为 patch 嵌入，添加位置编码，应用 Transformer 编码器，并生成上下文嵌入。
        # 输入：x，形状为 [batch_size, channels, height, width]（例如，[batch_size, 1, 1024, 128]，表示梅尔频谱图）。
        # 输出：contextual_emb，形状为 [batch_size, num_patches+1, embed_dim]（例如，[batch_size, 513, 768]，包含 CLS token 和 patch 嵌入）。
        # x(batch_size,1,1024,128)

        x = self.patch_embed(x)  #将输入梅尔频谱图分割为 patch，并映射到嵌入空间。  x(bs,512,768)，768为特征通道数，512为patch数。

        # add pos embed w/o cls token
        # self.pos_embed 是预定义的位置编码张量，形状为 [1, num_patches+1, embed_dim]
        # 第一个位置（self.pos_embed[:, 0, :]）是为 CLS token 保留的。
        # 其余位置（self.pos_embed[:, 1:, :]）对应于 patch 的位置编码，形状为 [1, 512, 768]。
        # x + self.pos_embed[:, 1:, :] 将每个 patch 的嵌入与对应的位置编码相加，保持形状 [batch_size, 512, 768]。
        x = x + self.pos_embed[:, 1:, :]   #为 patch 嵌入添加位置编码，不包括 CLS token 的位置编码

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)


        # append cls token
        # 添加 CLS token 到 patch 嵌入序列，形成完整的 Transformer 输入。
        # [CLS] Token 被用来汇总patch的全局特征  在 ViT 中，输入图像被划分为多个固定大小的 Patch，
        # 然后每个 Patch 被投影到一个 D维的特征向量空间，形成 Token 序列。
        # 为了让模型学习到全局信息，ViT 在输入序列的最前面添加一个特殊的 [CLS] Token：
        # CLS token的最终表示会包含输入音频序列的全局信息。由于自注意力机制的设计，
        # 它会在不同的时间步长上与其他音频帧（或其他音频标记）进行交互，最终生成一个综合的音频表示。
        # 在音频重建任务中，模型的目标是根据这个全局表示来完成对音频的恢复或生成。

        # self.cls_token 是一个可学习的 CLS token，形状为 [1, 1, embed_dim]（例如，[1, 1, 768]），用于捕获全局上下文。
        # self.pos_embed[:, :1, :] 是 CLS token 的位置编码，形状为 [1, 1, 768]。
        cls_token = self.cls_token + self.pos_embed[:, :1, :]  #将 CLS token 与其位置编码相加，形状仍为 [1, 1, 768]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #扩展 CLS token 以匹配批次大小，形状变为 [batch_size, 1, 768]。
        x = torch.cat((cls_tokens, x), dim=1)   # torch.cat((cls_tokens, x), dim=1) 将 CLS token 拼接在 patch 嵌入序列的前面：
        # CLS token 插入到音频序列的开头，作为输入的一部分。这样，模型在经过自注意力计算时，CLS token 会与其他音频帧的特征进行交互，
        # 最终得到一个包含全局信息的 CLS token 表示。
        # x 形状变为 [batch_size, 513, 768]，表示包含 CLS token 的完整序列。

        # apply Transformer blocks
        # 通过 Transformer 编码器块处理输入序列，捕获上下文信息，并收集深层嵌入。
        contextual_embs = []
        for n, blk in enumerate(self.blocks):
            # 每个 Block 的结构：LayerNorm（norm1 和 norm2）：在每个子层之前进行层归一化，以帮助训练过程中保持稳定。
            # Self-Attention（attn）：自注意力机制，模型在每个位置上计算输入序列中各个部分的相关性（Q、K、V）。这使得模型能够捕捉长范围的依赖关系。
            #   qkv: 线性层，用于计算查询（Q）、键（K）和值（V），它们会根据注意力机制来进行加权组合。
                # 通过计算 Q 和 K 的点积，得到注意力分数，再通过Softmax对其进行归一化，得到每个位置对其他位置的关注度，最后将关注度与V进行加权求和
            #   proj: 线性层，用于将注意力的输出映射回原始的嵌入维度（768），并进行投影。
            # MLP：每个 Block 包含一个简单的多层感知机（包括两个线性层和一个 GELU 激活函数）。
            # Dropout：每个模块都有 Dropout，防止过拟合。
            # Drop Path：用来在训练过程中随机丢弃某些路径，这是一种正则化技术，有助于提高模型的泛化能力。

            x = blk(x)  # x = blk(x) 依次通过每个 Transformer 块，更新 x 的表示，形状保持 [batch_size, 513, 768]。
            if n > self.contextual_depth:  # 如果 self.contextual_depth = 8，则从第 9 层以后（索引从 0 开始）的输出会被收集。
                contextual_embs.append(self.norm(x))  #self.norm 是一个 LayerNorm 模块，标准化每个 token 的嵌入（[batch_size, 513, 768]）。
        # x = self.norm(x)
        contextual_emb = torch.stack(contextual_embs, dim=0).mean(dim=0)   # 收集的深层嵌入堆叠并取平均，生成最终的上下文嵌入。
        #contextual_embs 是一个列表，包含若干 [batch_size, 513, 768] 的张量（例如，如果 self.contextual_depth = 8 和总层数为 12，则包含 4 个张量）。
        return contextual_emb

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        if self.decoder_mode != 0:
            B, L, D = x.shape
            x = x[:, 1:, :]
            if self.use_custom_patch:
                x = x.reshape(B, 101, 12, D)
                x = torch.cat([x, x[:, -1, :].unsqueeze(1)], dim=1)  # hack
                x = x.reshape(B, 1224, D)
        if self.decoder_mode > 3:  # mvit
            x = self.decoder_blocks(x)
        else:
            # apply Transformer blocks
            for blk in self.decoder_blocks:
                x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        if self.decoder_mode != 0:
            if self.use_custom_patch:
                pred = pred.reshape(B, 102, 12, 256)
                pred = pred[:, :101, :, :]
                pred = pred.reshape(B, 1212, 256)
            else:
                pred = pred
        else:
            pred = pred[:, 1:, :]
        return pred, None, None  # emb, emb_pixel

    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.8):
        emb_enc, mask, ids_restore, _ = self.forward_encoder(
            imgs, mask_ratio, mask_2d=self.mask_2d
        )
        pred, _, _ = self.forward_decoder(emb_enc, ids_restore)  # [N, L, p*p*3]
        loss_recon = self.forward_loss(
            imgs, pred, mask, norm_pix_loss=self.norm_pix_loss
        )
        loss_contrastive = torch.FloatTensor([0.0]).cuda()
        return loss_recon, pred, mask, loss_contrastive


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
