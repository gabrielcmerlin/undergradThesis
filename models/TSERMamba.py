import torch
from mamba_ssm import Mamba
from models.RevIN import RevIN
import copy
from einops.layers.torch import Rearrange
import numpy as np
import joblib

class TSERMamba(torch.nn.Module):
    def __init__(self,
                 enc_in,
                 seq_len,
                 projected_space,
                 rescale_size,
                 patch_size,
                 dropout,
                 initial_focus,
                 d_state,
                 dconv,
                 e_fact,
                 num_mambas,
                 flip_dir,
                 output_dim=1):
        """
        Modelo TSERMamba (Time Series Extrinsic Regression)

        Args:
            enc_in: número de canais de entrada (ex: 1)
            seq_len: comprimento da série temporal
            projected_space: dimensão do espaço projetado
            rescale_size: tamanho da imagem (CWT)
            patch_size: tamanho do patch 2D
            dropout: taxa de dropout
            initial_focus: valor inicial do parâmetro learnable_focus
            d_state: dimensão de estado interna do Mamba
            dconv: tamanho da convolução no Mamba
            e_fact: fator de expansão no Mamba
            num_mambas: número de blocos Mamba
            flip_dir: dimensão usada para flip (ex: 1)
            output_dim: número de saídas (regressão extrínseca → geralmente 1)
        """

        super(TSERMamba, self).__init__()

        # Conversão da entrada CWT (imagem) para embedding temporal
        self.patcher = ConversionLayer(
            d_in=enc_in,
            l_in=seq_len,
            l_out=projected_space,
            d_out=enc_in,
            im_size=rescale_size,
            patch_size=patch_size
        )

        # Projeção Rocket / feature-level
        self.projector = torch.nn.Linear(seq_len, projected_space // 2)

        self.ln = torch.nn.LayerNorm(projected_space * 3)
        self.dropout = torch.nn.Dropout(dropout)
        self.learnable_focus = torch.nn.Parameter(torch.tensor([initial_focus]))
        self.gelu = torch.nn.GELU()

        # Módulos Mamba
        self.mamba1 = torch.nn.ModuleList([
            Mamba(d_model=projected_space * 3,
                  d_state=d_state,
                  d_conv=dconv,
                  expand=e_fact)
            for _ in range(num_mambas)
        ])
        self.mamba2 = torch.nn.ModuleList([
            Mamba(d_model=enc_in,
                  d_state=d_state,
                  d_conv=dconv,
                  expand=e_fact)
            for _ in range(num_mambas)
        ])

        self.flatten = torch.nn.Flatten(start_dim=1)

        # Camada final de regressão
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(projected_space * 3, (projected_space * 3) // 2),
            torch.nn.Dropout(dropout),
            torch.nn.Linear((projected_space * 3) // 2, output_dim)
        )

        # Guardar parâmetros úteis para forward
        self.num_mambas = num_mambas
        self.flip_dir = flip_dir
        self.projected_space = projected_space

    def regression(self, x_cwt, batch_x_features):

        x_projected = torch.cat([
            batch_x_features[:, :, :self.projected_space // 2],
            self.projector(batch_x_features[:, :, self.projected_space:])
        ], dim=2)

        # --- Extração de embeddings ---
        x_patched = self.patcher(x_cwt)
        
        # --- Fusão entre features Rocket e CWT ---
        x_fused = (self.learnable_focus * x_projected) + (2.0 - self.learnable_focus) * x_patched

        x_fused = self.gelu(x_fused)
        concatenated_x = torch.cat([x_patched, x_fused, x_projected], dim=2)
        concatenated_x = self.ln(concatenated_x)

        # --- Passagem pelos blocos Mamba ---
        x1 = concatenated_x.clone()
        for i in range(self.num_mambas):
            x1 = self.mamba1[i](x1) + x1.clone()

        # Direção reversa
        concatenated_x = torch.flip(concatenated_x, dims=[self.flip_dir])
        x1_flipped = concatenated_x.clone()
        for i in range(self.num_mambas):
            x1_flipped = self.mamba1[i](x1_flipped) + x1_flipped.clone()
        x1 = x1 + x1_flipped
        concatenated_x = torch.flip(concatenated_x, dims=[self.flip_dir])

        concatenated_x = torch.permute(concatenated_x, (0, 2, 1))
        x2 = concatenated_x.clone()
        for i in range(self.num_mambas):
            x2 = self.mamba2[i](x2) + x2.clone()
        x2 = torch.permute(x2, (0, 2, 1))

        # Caminho reverso
        x2_flipped = torch.flip(concatenated_x.clone(), dims=[self.flip_dir])
        for i in range(self.num_mambas):
            x2_flipped = self.mamba2[i](x2_flipped) + x2_flipped.clone()
        x2_flipped = torch.permute(x2_flipped, (0, 2, 1))
        x2 = x2 + x2_flipped
        x3 = x2 + x1

        # --- Pooling e regressão ---
        x3 = x3.mean(1)
        x3 = self.flatten(x3)
        x_out = self.regressor(x3)
        return x_out

    def forward(self, x_cwt, x_features):
        return self.regression(x_cwt, x_features)

class ConversionLayer(torch.nn.Module):
    def __init__(self, d_in, l_in, d_out, l_out, im_size, patch_size):
        super(ConversionLayer, self).__init__()
        self.d_in = d_in
        self.l_in = l_in
        self.d_out = d_out
        self.l_out = l_out
        self.im_size = im_size
        self.patch_size = patch_size

        self.patch_embedding = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=self.d_in,
                out_channels=self.d_out,
                kernel_size=(self.patch_size, self.patch_size),
                stride=(self.patch_size, self.patch_size),
                padding=0
            ),
            Rearrange('b c h w -> b c (h w)')
        )
        self.projector = torch.nn.Linear(
            (self.im_size // self.patch_size) * (self.im_size // self.patch_size),
            self.l_out
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.projector(x)
        return x