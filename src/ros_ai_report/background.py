import torch
import torchaudio

class BackgroundNoise(torch.nn.Module):
    def __init__(self, path, percentage):
        super().__init__()
        self.path = path
        self.percentage = percentage

    def forward(self, waveform):
        noise_waveform, _ = torchaudio.load(self.path)
        noise_waveform = noise_waveform[:,:waveform.size()[1]]
        waveform = (1.0 - self.percentage) * waveform + self.percentage * noise_waveform
        return waveform