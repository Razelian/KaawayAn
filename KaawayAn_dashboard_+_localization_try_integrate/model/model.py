"""DroneClassifier — CRNN for binary drone audio detection.

Architecture: CNN feature pyramid (3 blocks) → BiGRU bottleneck → FC head.
Input : log-mel spectrogram (B, 1, 64, T)
Output: raw logit (B, 1)  — pass through sigmoid for probability

credits: https://huggingface.co/AntoineNaccache/drone-audio-detector
"""
import torch
import torch.nn as nn
import torchaudio.transforms as T
from pathlib import Path
import numpy as np


# weights live next to this file (anchored so imports work from any cwd)
ckpt_path = Path(__file__).parent / "drone_classifier_aug_mixed.pt"

if not ckpt_path.is_file():
    raise FileNotFoundError(f"Model weights not found: {ckpt_path}")


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SharedEncoder(nn.Module):
    GRU_INPUT = 128 * 8
    GRU_HIDDEN = 128

    def __init__(self):
        super().__init__()
        self.enc1 = _conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = _conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = _conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.gru = nn.GRU(
            self.GRU_INPUT, self.GRU_HIDDEN,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.pool3(self.enc3(x))
        B, C, F, Tp = x.shape
        rnn_in = x.permute(0, 3, 1, 2).reshape(B, Tp, C * F)
        rnn_out, _ = self.gru(rnn_in)
        return rnn_out


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, rnn_out: torch.Tensor) -> torch.Tensor:
        return self.fc(rnn_out.mean(dim=1))


class DroneClassifier(nn.Module):
    """Full model: encoder + classifier head. Call predict() with int16 samples."""

    def __init__(self):
        super().__init__()
        self.encoder = SharedEncoder()
        self.classifier = ClassifierHead()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16_000, n_fft=512, hop_length=160,
            n_mels=64, f_min=50, f_max=5_500,
        )
        self.load_state()

    def load_state(self):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        # strict=False: ignores separator.* keys from joint-training checkpoints
        self.load_state_dict(state, strict=False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))

    def predict(self, samples: np.ndarray) -> float:
        arr = np.asarray(samples, dtype=np.float32).ravel()[:16_000] / 32768.0
        waveform = torch.from_numpy(arr).unsqueeze(0)  # (1, N)
        log_mel = (T.AmplitudeToDB()(self.mel_transform(waveform)) + 40) / 40
        x = log_mel.unsqueeze(0)  # (1, 1, 64, T)
        with torch.no_grad():
            prob = torch.sigmoid(self(x)).item()
        return prob


def load_classifier(checkpoint: str, device: torch.device | str = "cpu") -> DroneClassifier:
    model = DroneClassifier()
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


# Back-compat alias
JointModel = DroneClassifier
