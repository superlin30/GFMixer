# config/FAT_configs.py

from types import SimpleNamespace


def get_FAT_config(
    d_model: int,
    n_heads: int,
    max_sequence_length: int = 720,
    fourier_dim: int = 64,
    fourier_learnable: bool = False,
    rope_theta: int = 10000,
    init_device: str = "cpu"
) -> SimpleNamespace:
    """
    Returns configuration for Fourier-Aware Positional Encoding (FOPE)
    Used in FAB module of GFMixer
    """

    return SimpleNamespace(
                d_model=d_model,
                n_heads=n_heads,
                max_sequence_length=720,
                rope=True,
                rope_theta=10000,
                rope_init_distribution="exponential",
                rope_no_pos=False,
                rope_no_rotary=False,
                rope_no_repetition=False,
                rope_full_precision=False,
                rope_learnable=False,
                rope_include_neg_freq=False,
                rope_init_upper_freq=1.0,
                rope_init_floor_freq=0.0,
                rope_floor_freq_ratio=0.1,
                rope_upper_freq_ratio=0.8,
                rope_clamp_floor_freq=True,
                rope_clamp_floor_to_zero=True,
                rope_clamp_upper_freq=False,
                rope_clamp_upper_to_zero=False,
                rope_zero_freq_ratio=0.0,
                rope_clamp_to_linear=False,
                rope_clamp_to_linear_mode='floor',
                len_extra=False,
                len_extra_type="PI",
                len_extra_before_clamp=False,
                len_extra_orig_length=0,
                len_extra_yarn_scale=16.0,
                len_extra_yarn_beta_fast=32.0,
                len_extra_yarn_beta_slow=1.0,
                len_extra_yarn_factor=1.0,
                fourier=True,
                fourier_dim=64,
                fourier_init="eye_xavier_norm",
                fourier_init_norm_gain=0.3,
                fourier_separate_basis=True,
                fourier_separate_head=True,
                fourier_learnable=False,
                fourier_norm=False,
                fourier_ignore_zero=True,
                init_device="cpu",
                )
