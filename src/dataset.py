import cv2
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class MultiModalSyncDataset(Dataset):
    def __init__(self, video_path, audio_path, target_dim=96):
        self.target_dim = target_dim
        self.frames = []
        self.masked_frames = []

        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (target_dim, target_dim))
            frame_normalized = frame_resized.astype(np.float32) / 255.0

            masked_frame = frame_normalized.copy()
            masked_frame[int(target_dim * 0.55) :, :] = 0.0

            self.frames.append(frame_normalized)
            self.masked_frames.append(masked_frame)
        cap.release()

        y, sr = librosa.load(audio_path, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)
        self.mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        self.length = min(len(self.frames), self.mel_spec.shape[1])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        orig_tensor = torch.tensor(self.frames[idx]).permute(2, 0, 1)
        masked_tensor = torch.tensor(self.masked_frames[idx]).permute(2, 0, 1)

        audio_feature = torch.tensor(self.mel_spec[:, idx]).float().unsqueeze(0)

        return masked_tensor, orig_tensor, audio_feature
