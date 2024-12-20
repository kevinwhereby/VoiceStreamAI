# isort: skip_file

from src.buffering_strategy.buffering_strategy_factory import (
    BufferingStrategyFactory,
)
from src.transcriber.transcriber import Transcriber


class Client:
    """
    Represents a client connected to the VoiceStreamAI server.

    This class maintains the state for each connected client, including their
    unique identifier, audio buffer, configuration, and a counter for processed
    audio files.

    Attributes:
        client_id (str): A unique identifier for the client.
        buffer (bytearray): A buffer to store incoming audio data.
        config (dict): Configuration settings for the client, like chunk length
                       and offset.
        file_counter (int): Counter for the number of audio files processed.
        total_samples (int): Total number of audio samples received from this
                             client.
        sampling_rate (int): The sampling rate of the audio data in Hz.
        samples_width (int): The width of each audio sample in bits.
    """

    def __init__(self, client_id, sampling_rate, samples_width):
        self.client_id = client_id
        self.buffer = bytearray()
        self.scratch_buffer = bytearray()
        self.config = {
            "language": None,
            "processing_strategy": "silence_at_end_of_chunk",
            "processing_args": {
                "chunk_length_seconds": 2,
                "chunk_offset_seconds": 0.1,
            },
        }
        self.sampling_rate = sampling_rate
        self.samples_width = samples_width
        self.buffering_strategy = BufferingStrategyFactory.create_buffering_strategy(
            self.config["processing_strategy"],
            self,
            Transcriber(
                "http://ec2-54-194-146-152.eu-west-1.compute.amazonaws.com:8080/v1/audio/transcriptions"
            ),
            **self.config["processing_args"],
        )

    def update_config(self, config_data):
        self.config.update(config_data)
        self.buffering_strategy = BufferingStrategyFactory.create_buffering_strategy(
            self.config["processing_strategy"],
            self,
            Transcriber(
                "http://ec2-54-194-146-152.eu-west-1.compute.amazonaws.com:8080/v1/audio/transcriptions"
            ),
            **self.config["processing_args"],
        )

    def append_audio_data(self, audio_data):
        self.buffer.extend(audio_data)

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        self.buffering_strategy.process_audio(websocket, vad_pipeline)
