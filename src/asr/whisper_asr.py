import os

import torch
from transformers import pipeline
import numpy as np

from src.audio_utils import save_audio_to_file
from src.client import Client

from .asr_interface import ASRInterface


class WhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = kwargs.get("model_name", "openai/whisper-large-v3")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    async def transcribe(self, client: Client):
        to_return = ""

        audio_data = np.frombuffer(client.scratch_buffer, dtype=np.int16).astype(np.float32)
        audio_data = audio_data / 32768.0  # Normalize to [-1, 1] range
        print(f"Audio shape: {audio_data.shape}")
        print(f"Audio min/max: {audio_data.min()}, {audio_data.max()}")
        print(f"Audio data length: {len(audio_data)}")
        print(f"Pipeline config: {self.asr_pipeline.model.config}")

        for item in self.asr_pipeline(audio_data):
            print(f"Got: {item}, {type(item)}")
            to_return += item

        to_return = {
            "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            "language_probability": None,
            "text": to_return.strip(),
            "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return
