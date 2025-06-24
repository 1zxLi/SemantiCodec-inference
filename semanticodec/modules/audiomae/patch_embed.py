import torch
import torch.nn as nn
from timm.models.layers import to_2tuple


class PatchEmbed_org(nn.Module):
    """Image to Patch Embedding
    功能：PatchEmbed_org 将输入的 2D 数据（例如，梅尔频谱图）分割为非重叠的 patch，并通过 2D 卷积生成嵌入向量，供 Transformer 模型处理。
    输入：通常是形状为 [batch_size, channels, height, width] 的张量，例如 [batch_size, 1, 1024, 128]（梅尔频谱图，单通道，1024 个时间步，128 个频率维度）。
    输出：形状为 [batch_size, num_patches, embed_dim] 的张量，例如 [batch_size, 512, 768]（512 个 patch，每个 patch 嵌入维度为 768）。
    用途：在 AudioMAE 模型（如 mae_vit_base_patch16）中，该模块是编码器的第一步，将输入转换为 Transformer 可处理的序列格式。

    patch 是将输入数据（例如图像或梅尔频谱图）分割成的小块（sub-regions），每个小块被独立处理并映射到高维嵌入空间，供模型进一步分析。
    Patch 是输入数据（通常是 2D 的图像、视频帧或梅尔频谱图）被分割成的一个个小区域（通常是矩形或正方形）。
    其分割方式可以是非重叠（如 PatchEmbed_org）或重叠（如 PatchEmbed_new），具体取决于步幅（stride）设置。
    stride = patch_size：非重叠（PatchEmbed_org）。
    stride < patch_size：重叠（PatchEmbed_new）。
    """


    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 定义 2D 卷积层，将每个 patch 映射到嵌入空间。
        # 这个卷积的输入通道为 1 输出通道为 768， 即768个卷积核，每个核的大小为16，16，且步长也为16，即卷积时，不重叠

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # 应用卷积层，将输入分割为 patch 并映射到嵌入空间。 效果：每个 16×16 的 patch 被映射为一个 768 维的向量，空间分辨率从 (1024, 128) 降为 (64, 8)。
        y = x.flatten(2).transpose(1, 2)  # 将卷积输出展平并转置，转换为 Transformer 可处理的序列格式。
        # 768种描述（嵌入），每种描述下有512个patch  需要注意的是512个patch是由16*16的卷积尺寸（视野）裁切的。因此该维度被称为patch
        return y


class PatchEmbed_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )  # with overlapped patches
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        # self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size)  # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h * w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1, 1, img_size[0], img_size[1])).shape

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)  # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2)  # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2)  # 32, 768, 1212 -> 32, 1212, 768
        return x


class PatchEmbed3D_new(nn.Module):
    """Flexible Image to Patch Embedding"""

    def __init__(
        self,
        video_size=(16, 224, 224),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=768,
        stride=(2, 16, 16),
    ):
        super().__init__()

        self.video_size = video_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride
        )
        _, _, t, h, w = self.get_output_shape(video_size)  # n, emb_dim, h, w
        self.patch_thw = (t, h, w)
        self.num_patches = t * h * w

    def get_output_shape(self, video_size):
        # todo: don't be lazy..
        return self.proj(
            torch.randn(1, self.in_chans, video_size[0], video_size[1], video_size[2])
        ).shape

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.proj(x)  # 32, 3, 16, 224, 224 -> 32, 768, 8, 14, 14
        x = x.flatten(2)  # 32, 768, 1568
        x = x.transpose(1, 2)  # 32, 768, 1568 -> 32, 1568, 768
        return x


if __name__ == "__main__":
    # patch_emb = PatchEmbed_new(img_size=224, patch_size=16, in_chans=1, embed_dim=64, stride=(16,16))
    # input = torch.rand(8,1,1024,128)
    # output = patch_emb(input)
    # print(output.shape) # (8,512,64)

    patch_emb = PatchEmbed3D_new(
        video_size=(6, 224, 224),
        patch_size=(2, 16, 16),
        in_chans=3,
        embed_dim=768,
        stride=(2, 16, 16),
    )
    input = torch.rand(8, 3, 6, 224, 224)
    output = patch_emb(input)
    print(output.shape)  # (8,64)
