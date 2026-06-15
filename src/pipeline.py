import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import subprocess
from torch.utils.data import DataLoader
from src.audio_utils import process_audio_pipeline
from src.dataset import MultiModalSyncDataset
from src.model import LipSyncAutoencoder

def run_sync_india_pipeline(video_path, target_lang="hi", epochs=5, status_callback=None):
    if status_callback: status_callback("🔊 Extracting & translating multi-modal tracks...")
    translated_wav = process_audio_pipeline(video_path, target_lang=target_lang)
    
    if status_callback: status_callback("📊 Initializing dynamic tensor dataset mapping...")
    dataset = MultiModalSyncDataset(video_path, translated_wav)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    if status_callback: status_callback("🧠 Constructing Neural Network Layers...")
    model = LipSyncAutoencoder()
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Active Person-Specific Optimization Loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for masked_face, orig_face, audio_feat in dataloader:
            optimizer.zero_grad()
            predictions = model(masked_face, audio_feat)
            loss = criterion(predictions, orig_face)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if status_callback: 
            status_callback(f"🏋️ Training Viseme Layers | Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    # Inference State - Reconstruct final synchronized frames
    if status_callback: status_callback("🎬 Synthesizing localized output frames...")
    model.eval()
    temp_output_avi = os.path.join("outputs", "temp_render.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(temp_output_avi, fourcc, 25.0, (96, 96))
    
    with torch.no_grad():
        for masked_face, _, audio_feat in dataloader:
            predictions = model(masked_face, audio_feat)
            for pred in predictions:
                img = pred.permute(1, 2, 0).numpy() * 255.0
                img = cv2.convertScaleAbs(img)
                out_video.write(img)
    out_video.release()
    
    # Multi-Modal Multiplexing - Mux the generated video frames with the translated audio track via FFmpeg
    final_output_mp4 = os.path.join("outputs", f"synchronized_{target_lang}.mp4")
    mux_command = f'ffmpeg -y -i "{temp_output_avi}" -i "{translated_wav}" -c:v libx264 -c:a aac -strict experimental "{final_output_mp4}"'
    subprocess.call(mux_command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    return final_output_mp4