import torch
import torch.nn as nn


class LipSyncAutoencoder(nn.Module):
    def __init__(self):
        super(LipSyncAutoencoder, self).__init__()

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Out: 48x48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Out: 24x24
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Out: 128x128
            nn.ReLU(),
        )

        self.audio_encoder = nn.Sequential(
            nn.Linear(80, 64), nn.ReLU(), nn.Linear(64, 128), nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Out: 24x24
            nn.ReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Out: 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 3, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # Out: 96x96
            nn.Sigmoid(),  # Bound pixel distributions strictly between 0.0 and 1.0
        )

    def forward(self, masked_face, audio_spec):
        face_feats = self.face_encoder(masked_face)

        audio_feats = self.audio_encoder(audio_spec.squeeze(1))  # shape: [B, 128]
        audio_feats = audio_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 12, 12)

        fused_latent = torch.cat(
            (face_feats, audio_feats), dim=1
        )  # shape: [B, 256, 12, 12]

        return self.decoder(fused_latent)
