import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from semanticodec.modules.audiomae.AudioMAE import Vanilla_AudioMAE
from vector_quantize_pytorch import VectorQuantize
from vector_quantize_pytorch import ResidualVQ
from semanticodec.utils import (
    concat_1x2,
    concat_2x2,
    PositionalEncoding,
    extract_kaldi_fbank_feature,
)


class AudioMAEConditionQuantResEncoder(nn.Module):
    def __init__(
        self,
        centroid_npy_path=None,
        feature_dimension=768,
        codebook_size=8192,
        codebook_dim=None,
        use_cosine_sim=False,
        decay=0.9,
        residual_encoder="lstm",
        lstm_layer=2,
        lstm_bidirectional=True,
        commitment_weight=1.0,
        rvq_layers=0,
        use_oracle=False,
        use_positional_embedding=True,
    ):
        super().__init__()
        self.use_oracle = use_oracle
        self.use_positional_embedding = use_positional_embedding
        self.residual_encoder = residual_encoder
        self.downsampling_rate = feature_dimension // 768
        self.feature_dimension = feature_dimension
        self.device = None
        self.pos_embed = PositionalEncoding(seq_length=512, embedding_dim=192)

        assert centroid_npy_path is not None, "centroid_npy_path is required"
        self.centroid_npy = torch.from_numpy(np.load(centroid_npy_path))

        self.centroid_npy.requires_grad = False
        self.audiomae = Vanilla_AudioMAE()
        self.audiomae.eval()
        for p in self.audiomae.parameters():
            p.requires_grad = False

        self.no_audiomae_mask = True
        self.no_audiomae_average = False

        if self.residual_encoder == "lstm":
            self.encoder = nn.LSTM(
                input_size=feature_dimension * 2,
                hidden_size=feature_dimension * 2,
                num_layers=lstm_layer,
                bias=True,
                batch_first=True,
                bidirectional=lstm_bidirectional,
            )
        else:
            raise ValueError("Invalid model name %s" % self.residual_encoder)

        self.encoder_output_linear = nn.Linear(
            in_features=feature_dimension * 2
            if not lstm_bidirectional
            else feature_dimension * 4,
            out_features=feature_dimension,
            bias=False,
        )

        self.rvq_layers = rvq_layers
        self.codebook_size = codebook_size
        if rvq_layers <= 0:
            self.quantizer = VectorQuantize(
                dim=feature_dimension,
                codebook_size=codebook_size,
                decay=decay,
                commitment_weight=commitment_weight,
                codebook_dim=codebook_dim,
                use_cosine_sim=use_cosine_sim,
            )
        else:
            self.quantizer = ResidualVQ(
                dim=feature_dimension,
                num_quantizers=rvq_layers,  # specify number of quantizers
                codebook_size=codebook_size,  # codebook size
            )

        self.indices_statistic_count = 0
        self.indices_statistic = {}
        self.eval()

    def mark_out_padding(self, feature, padding_cutoff_index):
        # 将填充部分特征值先置0，再置为-1， 相当于-1的全为填充部分
        feature_temporal_dim = feature.shape[-2]
        for i, index in enumerate(padding_cutoff_index):
            feature_cutoff_index = math.ceil(feature_temporal_dim * index)
            feature[i, int(feature_cutoff_index) :] *= 0.0
            feature[i, int(feature_cutoff_index) :] -= 1.0
        return feature

    # Required
    def get_unconditional_condition(self, batchsize):
        param = next(self.audiomae.parameters())
        assert param.requires_grad == False
        device = param.device
        token_num = 512
        representation_quant = (
            torch.zeros((batchsize, token_num, 768)).to(device).float()
        )
        if self.use_positional_embedding:
            pe = self.pos_embed(representation_quant)
            representation_quant = torch.cat(
                [representation_quant, pe.repeat(batchsize, 1, 1)], dim=-1
            )
        return [
            representation_quant,
            torch.ones((batchsize, token_num)).to(device).float(),
        ]

    def quant_mem_efficient(
        self, representation, first_token_removed=False, feature_dim=768
    ):
        # representation（1，128，1024）
        assert representation.size(-1) % 768 == 0
        # Removing the first token and keeping the shape as [batch_size, seq_length - 1, 768] for clarity
        # 去掉第一维的CLS token
        if not first_token_removed:
            representation = representation[
                :, 1:, :
            ]  # Shape: [batch_size, seq_length - 1, 768]

        # Compute squared norms of each row in representation
        # 计算每个 token 的平方范数：
        norm_rep = representation.pow(2).sum(
            dim=2, keepdim=True
        )  # Shape: [batch_size, seq_length - 1, 1]

        # Compute squared norms of centroids
        # 计算质心的平方范数：
        norm_cent = self.centroid_npy.pow(2).sum(
            dim=1, keepdim=True
        )  # Shape: [2048, 1]

        # Compute dot products
        # Reshape representation for batch matrix multiplication: [batch_size * (seq_length - 1), 768]
        # 计算点积实际上是在度量每个 token 与每个质心之间的相似度。对于两个向量 a 和 b，它们的点积 a.b 越大，
        # 表示它们在向量空间中越相似（即它们的夹角越小，方向越一致）。
        # 张量展平
        rep_flat = representation.reshape(-1, feature_dim)
        # Dot product, need to transpose centroids: [batch_size * (seq_length - 1), 2048]
        # 计算 representation 中每个 token 与每个质心的点积
        #
        dot_product = torch.mm(rep_flat, self.centroid_npy.t())
        # 恢复原来的维度
        dot_product = dot_product.reshape(
            representation.shape[0], representation.shape[1], -1
        )  # Reshape back
        # 计算 L2 距离：  ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b  #a.b 为点积
        # Compute L2 distance using the formula: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        distances = norm_rep + norm_cent.t() - 2 * dot_product  # Correct broadcasting
        # 选择最近的质心：
        # Find the index of the closest centroid for each vector
        _, tokens = torch.min(distances, dim=2)  # Shape: [batch_size, seq_length - 1]

        return tokens

    def unquant(self, tokens):
        """
        Project the quantized tokens into continuous representation with self.centroid_npy.
        Args:
            tokens (torch.Tensor): The quantized tokens, shape [batch_size, seq_length - 1]
        Returns:
            torch.Tensor: The continuous representation, shape [batch_size, seq_length - 1, feature_dim]
        """
        return F.embedding(
            tokens, self.centroid_npy
        )  # Shape: [batch_size, seq_length - 1, 768]

    def indices_utilization_statistic(self, indices):
        # indices shape: [batchsize, 256, self.rvq_layers], values are integer codebook indices
        if indices.dim() == 2:
            indices = indices.unsqueeze(-1)

        # Update statistics with current indices
        batch_size, _, rvq_layers = indices.shape

        # Initialize the statistic data structure if not already done
        if not self.indices_statistic:
            # Create a list of dictionaries, one for each RVQ layer
            self.indices_statistic = [{} for _ in range(rvq_layers)]

        # Process each RVQ layer separately
        for layer in range(rvq_layers):
            layer_indices = (
                indices[:, :, layer].view(-1).cpu().tolist()
            )  # Flatten and convert to list for easy counting
            for idx in layer_indices:
                if idx in self.indices_statistic[layer]:
                    self.indices_statistic[layer][idx] += 1
                else:
                    self.indices_statistic[layer][idx] = 1

        # Update count and possibly calculate statistics
        if self.indices_statistic_count % 10000 == 0:
            # Calculate and print statistics for each codebook
            for layer, stats in enumerate(self.indices_statistic):
                utilization_rate = len(list(stats.keys())) / self.codebook_size
                utilizations = list(stats.values())
                print(
                    f"\n\nLayer {layer} Utilization Rate: {utilization_rate}",
                    "max utilization",
                    {max(utilizations)},
                    "min utilization",
                    {min(utilizations)},
                    "std",
                    np.std(utilizations),
                    "median",
                    np.median(utilizations),
                    "\n\n",
                )
                metrics = {
                    f"codec/{layer}_utilization": utilization_rate,
                    f"codec/{layer}_utilization_max": max(utilizations),
                    f"codec/{layer}_utilization_min": min(utilizations),
                    f"codec/{layer}_utilization_std": np.std(utilizations),
                    f"codec/{layer}_utilization_median": np.median(utilizations),
                }
                print("\n")
                print(metrics)
                print("\n")

            self.indices_statistic = [{} for _ in range(rvq_layers)]
            self.indices_statistic_count = 0

        self.indices_statistic_count += 1

    def concate(self, representation):
        assert representation.size(-1) == 768
        representation = representation[:, 1:, :].transpose(1, 2)  # 去掉第一个 token（通常是 [CLS]），因为通常这个 token 不参与后续的处理。
        bs, embedding_dim, token_num = representation.size()
        representation = representation.reshape(bs, embedding_dim, 64, 8).permute(
            0, 2, 3, 1
        )
        if self.downsampling_rate == 2:
            concatenated = concat_1x2(representation)
        elif self.downsampling_rate == 4:
            concatenated = concat_2x2(representation)
        else:
            raise ValueError("Invalid downsampling rate %s" % self.downsampling_rate)
        return concatenated  # [bs, token_num, embedding_dim]

    def get_unconditional_condition(self, batchsize):
        param = next(self.audiomae.parameters())
        assert param.requires_grad == False
        device = param.device
        token_num = 512 // self.downsampling_rate
        representation_quant = (
            torch.zeros((batchsize, token_num, self.feature_dimension))
            .to(device)
            .float()
        )
        if self.use_positional_embedding:
            pe = self.pos_embed(representation_quant)
            if not self.use_oracle:
                representation_quant = torch.cat(
                    [
                        representation_quant,
                        representation_quant,
                        pe.repeat(batchsize, 1, 1),
                    ],
                    dim=-1,
                )
            else:
                representation_quant = torch.cat(
                    [representation_quant, pe.repeat(batchsize, 1, 1)],
                    dim=-1,
                )
        return self.wrap_return_dict(
            crossattn_audiomae_pooled=[
                representation_quant,
                torch.ones((batchsize, token_num)).to(device).float(),
            ],
            commit_loss=torch.zeros((1,)).to(device),
        )

    def long_token_split_window(self, tokens, window_length=512, overlap=0.0625):
        # Overlap 0.64 seconds
        # batch: [batchsize, token_length, embedding_dimension]
        # Split into segments with overlap
        _, token_length, _ = tokens.size()
        overlap = int(window_length * overlap)
        current_start = 0
        token_window_list = []
        while current_start + window_length < token_length:
            current_batch = tokens[:, current_start : current_start + window_length, :]
            token_window_list.append(current_batch)
            current_start += window_length - overlap

        remaining_batch = tokens[:, current_start:, :]

        if remaining_batch.size(-2) > 0:
            # Pad to window length
            # remaining_batch = F.pad(remaining_batch, (0, 0, 0, window_length - remaining_batch.size(-2), 0, 0))
            token_window_list.append(remaining_batch)
        return token_window_list

    def forward(self, batch):
        #处理一批音频输入，生成量化 token。
        #输入：batch（形状为 [batch_size, seq_length, 128] 的张量，通常为梅尔频谱图）。
        #输出：拼接后的 token（[batch_size, total_tokens]）。
        # Perform padding before this function
        # Trim the audio token after this function
        assert batch.size(-1) == 128 and batch.size(-2) % 1024 == 0  #确保输入时间维度可被 1024 整除，特征维度为 128。
        if self.device is None:
            self.device = batch.device
            self.centroid_npy = self.centroid_npy.half()
            self.centroid_npy = self.centroid_npy.to(self.device)

        window_length = 1024
        current_start = 0
        total_length_batch = batch.size(-2)

        tokens_list = []
        quantized_feature_list = []
        #将输入分割为 1024 帧的窗口。对每个窗口调用 _forward 处理。拼接所有窗口的 token。
        while current_start + window_length <= total_length_batch:
            current_batch = batch[:, current_start : current_start + window_length, :]
            with torch.no_grad():
                # [bs, 513, 768]
                output = self._forward(current_batch)
                tokens_list.append(output["tokens"])
                quantized_feature_list.append(output["quantized_feature"])
            current_start += window_length
        return torch.cat(tokens_list, dim=1)

    def _forward(self, batch):
        """
        功能：单个音频窗口的核心前向传播。
        输入：batch（形状为 [batch_size, 1024, 128] 的张量）。
        输出：包含以下内容的字典：
        quantized_feature：条件表示和掩码。
        tokens：量化 token（oracle 模式下为 None）
        """
        assert batch.size(-2) == 1024 and batch.size(-1) == 128

        if self.device is None:
            self.device = batch.device
            self.centroid_npy = self.centroid_npy.half()
            self.centroid_npy = self.centroid_npy.to(self.device)

        batch = batch.unsqueeze(1)

        padding_cutoff_index = []
        temporal_dim = batch.shape[-2]
        for i in range(batch.shape[0]):  #在批次下对每条数据进行填充检测

            active_index = (
                torch.std(batch[i, 0], dim=-1) <= 1e-7   #batch[i, 0]：(1024，128)  active_index：(1024),对其每帧进行填充数据检测
            )  # F F T T F F T T T T T
            # If there are empty segment in the audio or there are padding in the audio
            try:
                if active_index.any():   #如果存在填充数据，进一步确定最后一个有效帧
                    # Convert boolean tensor to integer tensor where False becomes 0
                    # 原active_index中的False为有效，True为填充，active_index == False则将有效帧置为True，填充帧置为False
                    # int_tensor为[1024]，其中1为有效帧，0为填充帧
                    int_tensor = active_index == False
                    # Find indices where the tensor is False
                    false_indices = torch.nonzero(int_tensor, as_tuple=False).squeeze()  #使用 torch.nonzero 找到 int_tensor 中值为 1 的索引，即有效时间步的位置。
                    # Get the last index of False
                    # last_false_index = false_indices[-1].item() if false_indices.numel() > 0 else -1
                    if false_indices.numel() > 0:
                        last_false_index = false_indices[-1].item()  #确定最后一个有效时间步的索引。
                    else:
                        last_false_index = -1
                    column_max = last_false_index + 1
                # If there are no any empty segment in the audio
                else:
                    column_max = temporal_dim
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(false_indices)
                print(false_indices.numel())
                column_max = 0   #表示整个样本被视为填充（保守策略）。

            padding_cutoff_index.append(column_max / temporal_dim)

        with torch.no_grad():
            # [bs, 513, 768]
            #使用 AudioMAE 提取特征（[batch_size, 1024, 768]）。
            # batch：（1，1，1024，128）  mel特征
            representation = self.audiomae(
                batch,
                no_mask=self.no_audiomae_mask,
                no_average=self.no_audiomae_average,
            )
            #representation ([batch_size, 513, 768])
            if self.downsampling_rate != 1:
                representation = self.concate(representation)
                representation = (
                    representation.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
                )
            else:
                representation = representation[:, 1:, :]
        # 下采样到 [batch_size, 128, 3072]
        if not self.use_oracle:
            # Quantize the audiomae representation to tokens
            tokens = self.quant_mem_efficient(
                representation,
                first_token_removed=True,
                feature_dim=self.feature_dimension,
            )
            # 量化后token为[batch_size, 128]
            # Change the token back to the representations, which information losed
            representation_quant = self.unquant(tokens)  # 根据token 结合码本， 译码回连续特征
            audiomae_feature_after_quant = representation_quant.clone()  # 创建个副本
            representation_quant_stack_unquant = torch.cat(
                [representation, representation_quant], dim=-1
            )    #这行代码将原始表示 representation 和反量化后的表示 representation_quant 沿着最后一个维度（即特征维度）拼接在一起，

            # 根据填充标识，对量化特征进行处理，将填充部分全部置为-1
            representation_quant_stack_unquant = self.mark_out_padding(
                representation_quant_stack_unquant, padding_cutoff_index
            )

            # Use the encoder to extract extra information for conditioning
            if self.residual_encoder == "transformer":
                representation_residual = self.encoder(
                    representation_quant_stack_unquant.permute(0, 2, 1)
                ).permute(0, 2, 1)
            elif (
                self.residual_encoder == "lstm"
                or self.residual_encoder == "mamba"
                or self.residual_encoder == "ResidualLSTM"
            ):
                representation_residual = self.encoder(
                    representation_quant_stack_unquant
                )

            # If you use LSTM as encoder
            if type(representation_residual) == tuple:
                representation_residual = representation_residual[0]

            representation_residual = self.encoder_output_linear(
                representation_residual
            )
            # 即插即用的残差矢量量化器，返回量化输出、所有的量化索引和损失。 输入为(1,128,3072)  输出为(1,128)
            self.quantizer = self.quantizer.half()
            representation_residual = representation_residual.half()
            representation_residual_quant, indices, commit_loss = self.quantizer(
                representation_residual
            )
            # import ipdb; ipdb.set_trace()
            # assert torch.max(self.quantizer.get_output_from_indices(indices).reshape(1, 512, 768)-representation_residual_quant) <= 1e-5
            # 将原始的 tokens 和量化后的索引 indices 进行拼接。 新的 tokens，形状为 [batch_size, seq_length - 1, 2]
            tokens = torch.cat([tokens.unsqueeze(-1), indices.unsqueeze(-1)], dim=-1)
            representation_quant = torch.cat(
                [representation_residual_quant, representation_quant], dim=-1
            )
        else:
            # Oracle
            param = next(self.audiomae.parameters())
            assert param.requires_grad == False
            tokens = None

            representation_quant = torch.cat([representation], dim=-1)
        #同样，对量化矢量进行填充帧处理，
        representation_quant = self.mark_out_padding(
            representation_quant, padding_cutoff_index
        )
        # 如果采用了位置编码，将残差量化特征中也加入位置编码
        if self.use_positional_embedding:
            pe = self.pos_embed(representation_quant).to(representation_quant.device)
            representation_quant = torch.cat(
                [representation_quant, pe.repeat(representation_quant.size(0), 1, 1)],
                dim=-1,
            )

        return self.wrap_return_dict(
            crossattn_audiomae_pooled=[
                representation_quant,
                torch.ones((representation_quant.size(0), representation_quant.size(1)))
                .to(representation_quant.device)
                .float(),
            ],
            tokens=tokens,
        )

    def token_to_quantized_feature(self, tokens):
        # 将离散的 token（整数索引）解量化（dequantize）为连续的特征向量。
        semantic_tokens, acoustic_tokens = tokens[..., 0], tokens[..., 1]  #分解语义量化和残差量化
        semantic_feature = self.unquant(semantic_tokens)  #恢复 AudioMAE 的语义信息，表示音频的高级特征（例如，内容、语义）。
        token_num, feature_dim = semantic_feature.shape[-2], semantic_feature.shape[-1]
        acoustic_feature = self.quantizer.get_output_from_indices(
            acoustic_tokens
        ).reshape(1, token_num, feature_dim)  #根据残差矢量量化的码矢编号，转换为连续的码矢向量。
        return torch.cat([acoustic_feature, semantic_feature], dim=-1)  #沿特征维度（最后一个维度）拼接。

    def wrap_return_dict(self, crossattn_audiomae_pooled, tokens):
        return {"quantized_feature": crossattn_audiomae_pooled, "tokens": tokens}
