from configparser import NoSectionError
import torch
import torch.nn as nn
import os
import torchaudio
import math

from semanticodec.modules.encoder.encoder import AudioMAEConditionQuantResEncoder
from semanticodec.modules.decoder.latent_diffusion.models.ddpm import (
    extract_encoder_state_dict,
    overlap_add_waveform,
)
from semanticodec.config import get_config
from semanticodec.modules.decoder.latent_diffusion.util import instantiate_from_config
from semanticodec.utils import extract_kaldi_fbank_feature
from huggingface_hub import hf_hub_download

# Constants
SAMPLE_RATE = 16000
SEGMENT_DURATION = 10.24
MEL_TARGET_LENGTH = 1024
AUDIOMAE_PATCH_DURATION = 0.16
SEGMENT_OVERLAP_RATIO = 0.0625


class SemantiCodec(nn.Module):
    def __init__(
        self,
        token_rate,
        semantic_vocab_size,
        ddim_sample_step=50,
        cfg_scale=2.0,
        checkpoint_path = None,
        cache_path="pretrained",
        local_model_dir="/path/to/your/models",
    ):
        super().__init__()
        self.token_rate = token_rate
        self.stack_factor_K = 100 / self.token_rate
        # stack_factor_K: 它的含义为audioMAE中 patch 到 tokens 的压缩率，其值为  100 / self.token_rate
        # token_rate就是每秒音频多少个tokens，而patch则是固定的每秒50个（10.24s/512个），
        # 而每个tokens包含两种结果：一是audioMAE，另一是残差矢量，因此stack_factor_K = （50 / self.token_rate） * 2
        # 乘 2 是因为每个tokens包含两种结果。因此压缩率 * 2

        self.ddim_sample_step = ddim_sample_step
        self.cfg_scale = cfg_scale

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")


        # Initialize encoder and decoder
        config, checkpoint_path, feature_dim, lstm_layers, semanticodebook = get_config(
            token_rate, semantic_vocab_size, checkpoint_path
        )
        # checkpoint_path = semanticodec_tokenrate_50
        encoder_checkpoint_path = os.path.join(checkpoint_path, "encoder.ckpt")
        if not os.path.exists(encoder_checkpoint_path):
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
                print(f"checkpoint cache dir '{cache_path}' was created.")
            encoder_checkpoint_path = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=checkpoint_path+"/encoder.ckpt",cache_dir=cache_path)
        decoder_checkpoint_path = os.path.join(checkpoint_path, "decoder.ckpt")
        if not os.path.exists(decoder_checkpoint_path):
            decoder_checkpoint_path = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=checkpoint_path+"/decoder.ckpt",cache_dir=cache_path)

        if not os.path.exists(semanticodebook):
            semanticodebook = "/".join(semanticodebook.split("/")[-3:])
            semanticodebook = hf_hub_download(repo_id="haoheliu/SemantiCodec",filename=semanticodebook,cache_dir=cache_path)

        # Initialize encoder
        print("🚀 Loading SemantiCodec encoder")
        state_dict = torch.load(encoder_checkpoint_path, map_location="cpu")
        self.encoder = AudioMAEConditionQuantResEncoder(
            feature_dimension=feature_dim,
            lstm_layer=lstm_layers,
            centroid_npy_path=semanticodebook,
        )
        self.encoder.load_state_dict(state_dict)
        self.encoder = self.encoder.half()
        self.encoder = self.encoder.to(self.device)
        print("✅ Encoder loaded")

        # Initialize decoder
        print("🚀 Loading SemantiCodec decoder")
        self.decoder = instantiate_from_config(config["model"])
        checkpoint = torch.load(decoder_checkpoint_path, map_location="cpu")
        self.decoder.load_state_dict(checkpoint)
        self.decoder = self.decoder.half()
        self.decoder = self.decoder.to(self.device)
        print("✅ Decoder loaded")

    def load_audio(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} does not exist")

        assert isinstance(filepath, str)
        waveform, sr = torchaudio.load(filepath)
        # resample to 16000
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            sr = SAMPLE_RATE
        # if stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1]
        # Calculate the original duration
        original_duration = waveform.shape[1] / sr
        # This is to pad the audio to the multiplication of 0.16 seconds so that the original audio can be reconstructed
        # 为什么要求音频时长为0.16S整数倍？  因为在patch提取过程中使用到的卷积核和步长均为(16, 16)，
        # 其感知野为16帧mel，每帧mel为10ms，因此需要音频时长必须为0.16s的整数倍，否则会出现不完整的 patch。
        original_duration = original_duration + (
            AUDIOMAE_PATCH_DURATION - original_duration % AUDIOMAE_PATCH_DURATION
        )
        # Calculate the token length in theory
        # stack_factor_K: 它的含义为audioMAE中 patch 到 tokens 的压缩率，其值为  100 / self.token_rate
        # token_rate就是每秒音频多少个tokens，而patch则是固定的每秒50个（10.24s/512个），
        # 而每个tokens包含两种结果：一是audioMAE，另一是残差矢量，因此stack_factor_K = （50 / self.token_rate） * 2
        # 乘 2 是因为每个tokens包含两种结果。因此压缩率 * 2
        target_token_len = (
            8 * original_duration / AUDIOMAE_PATCH_DURATION / self.stack_factor_K
        )
        segment_sample_length = int(SAMPLE_RATE * SEGMENT_DURATION)
        # Pad audio to the multiplication of 10.24 seconds for easier segmentations
        # 为什么要扩展到10.24s的整数倍？
        # audioMAE的输入为(bs,1024,128)的mel，每次必须要输入1024帧mel，也就是10.24s的音频。
        if waveform.shape[1] % segment_sample_length < segment_sample_length:
            waveform = torch.cat(
                [
                    waveform,
                    torch.zeros(
                        1,
                        int(
                            segment_sample_length
                            - waveform.shape[1] % segment_sample_length
                        ),
                    ),
                ],
                dim=1,
            )

        mel_target_length = MEL_TARGET_LENGTH * int(
            waveform.shape[1] / segment_sample_length
        )
        # Calculate the mel spectrogram
        mel = extract_kaldi_fbank_feature(
            waveform, sr, target_length=mel_target_length
        )["ta_kaldi_fbank"].unsqueeze(0)
        mel = mel.squeeze(1)
        assert mel.shape[-1] == 128 and mel.shape[-2] % 1024 == 0
        return mel, target_token_len

    def encode(self, filepath):
        mel, target_token_len = self.load_audio(filepath)
        mel = mel.half()
        tokens = self.encoder(mel.to(self.device))
        tokens = tokens[:, : math.ceil(target_token_len), :]
        # tokens 为audioMAE的量化和残差矢量量化的拼接，其维度为(1,token，2),2即为两种量化结果
        return tokens

    def decode(self, tokens):
        # tokens 为audioMAE的量化和残差矢量量化的拼接，其维度为(1,token，2),2即为两种量化结果
        # 首先， 将所有tokens 重新分割为对应patch格式的数据段，
        # 例如如果stack_factor_K为4，也就是说，512个patch 对应的 为128个 token  （注意是 （1，128，2））
        # *** 512个patch为10.24s，如果stack_factor_K为4，即token_rete参数为25，也就是每秒25个token, 10.24s就有256个token,因此audioMAE和残差各占128token，
        # 因此如果stack_factor_K为4，那么tokens为(1,token，2),windowed_token_list则包含了诸多(1,128，2)
        windowed_token_list = self.encoder.long_token_split_window(
            tokens,
            window_length=int(512 / self.stack_factor_K),  #stack_factor_K为audioMAE中 patch 到 tokens 的压缩率
            overlap=SEGMENT_OVERLAP_RATIO,
        )   #将 tokens（[1, token, 2]）分割为多个窗口（每个 [1, 128, 2]）。
        windowed_waveform = []
        for _, windowed_token in enumerate(windowed_token_list):
            # 遍历解码
            # 首先复现潜在特征latent 例如 [1, token, 6144]
            latent = self.encoder.token_to_quantized_feature(windowed_token)
            latent = torch.cat(
                [
                    latent,
                    torch.ones(
                        latent.shape[0],
                        int(512 / self.stack_factor_K) - latent.shape[1],
                        latent.shape[2],
                    ).to(latent.device)    #如果token不足10.24s， 这里仍然需要填充一下
                    * -1,
                ],
                dim=1,
            )
            latent = latent.half()
            # 将 latent 解码为波形
            waveform = self.decoder.generate_sample(
                latent,
                ddim_steps=self.ddim_sample_step,
                unconditional_guidance_scale=self.cfg_scale,
            )
            windowed_waveform.append(waveform)
        output = overlap_add_waveform(
            windowed_waveform, overlap_duration=SEGMENT_DURATION * SEGMENT_OVERLAP_RATIO
        )  #拼接多个窗口波形
        # Each patch step equal 16 mel time frames, which have 0.01 second
        trim_duration = (tokens.shape[1] / 8) * 16 * 0.01 * self.stack_factor_K
        return output[..., : int(trim_duration * SAMPLE_RATE)]

    def forward(self, filepath):
        with torch.cuda.amp.autocast():
            tokens = self.encode(filepath)
            #tokens = tokens.half()
            waveform = self.decode(tokens)
        return waveform


from thop import profile, clever_format
from torch.autograd import Variable
import torch

if __name__ == '__main__':
    # Initialize the SemantiCodec model
    net = SemantiCodec(
        token_rate=25,  # Example token rate
        semantic_vocab_size=4096,  # Example vocab size
        ddim_sample_step=50,
        cfg_scale=2.0,
        checkpoint_path = 'F:\Project\VoiceCodec\SemantiCodec\pretrained\SemantiCodec',  # Adjust as needed
        cache_path="pretrained",
        local_model_dir="/path/to/your/models"
    )

    # Prepare a sample input (file path to an audio file)
    sample_audio_path = r"F:\Project\VoiceCodec\SemantiCodec\testData\float16_output\16k-hybr.wav"  # Replace with a valid audio file path

    mel, target_token_len = net.load_audio(sample_audio_path)
    mel = mel.to(net.device).half()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()  # 设置为评估模式
    data_input = torch.randn(1, 10240, 128).to(device).half()
    # Profiling 编码器
    print("正在分析编码器...")
    flops_enc, params_enc = profile(net.encoder, inputs=(mel,))
    flops_enc, params_enc = clever_format([flops_enc, params_enc], "%.3f")
    print(f"编码器参数量: {params_enc}")
    print(f"编码器 FLOPs: {flops_enc}")

    # 获取 tokens
    tokens = net.encode(sample_audio_path)

    # Profiling 解码器
    print("正在分析解码器...")
    flops_dec, params_dec = profile(net.decoder, inputs=(tokens,))
    flops_dec, params_dec = clever_format([flops_dec, params_dec], "%.3f")
    print(f"解码器参数量: {params_dec}")
    print(f"解码器 FLOPs: {flops_dec}")

    # 总计
    total_params = float(params_enc[:-1]) + float(params_dec[:-1])  # 移除单位（M/G）
    total_flops = float(flops_enc[:-1]) + float(flops_dec[:-1])
    total_params, total_flops = clever_format([total_params * 1e6, total_flops * 1e9], "%.3f")
    print(f"总参数量: {total_params}")
    print(f"总 FLOPs: {total_flops}")