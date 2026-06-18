import torch
import torch.nn as nn

class LipSyncAutoencoder(nn.Module):
    def __init__(self):
        super(LipSyncAutoencoder, self).__init__()
        
        # High-Fidelity Spatial Face Encoder Matrix
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Out: 48x48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # Out: 24x24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # Out: 12x12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        # Enhanced Temporal Audio Encoder
        self.audio_encoder = nn.Sequential(
            nn.Linear(80, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2)
        )
        
        # High-Fidelity Generative Decoder Network - Restores micro-pixel details
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1), # Out: 24x24
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # Out: 48x48
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Out: 96x96
            nn.Sigmoid() 
        )

    def forward(self, masked_face, audio_spec):
        face_feats = self.face_encoder(masked_face) # Shape: [B, 256, 12, 12]
        
        audio_feats = self.audio_encoder(audio_spec.squeeze(1)) # Shape: [B, 256]
        audio_feats = audio_feats.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 12, 12)
        
        # Concat creates a thick, high-capacity latent representation [B, 512, 12, 12]
        fused_latent = torch.cat((face_feats, audio_feats), dim=1) 
        
        return self.decoder(fused_latent)