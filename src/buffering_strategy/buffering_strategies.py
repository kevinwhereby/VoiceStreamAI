import asyncio
import json
import os
import time

from .buffering_strategy_interface import BufferingStrategyInterface


class SilenceAtEndOfChunk(BufferingStrategyInterface):
    """
    A buffering strategy that processes audio at the end of each chunk with
    silence detection.

    This class is responsible for handling audio chunks, detecting silence at
    the end of each chunk, and initiating the transcription process for the
    chunk.

    Attributes:
        client (Client): The client instance associated with this buffering
                         strategy.
        chunk_length_seconds (float): Length of each audio chunk in seconds.
        chunk_offset_seconds (float): Offset time in seconds to be considered
                                      for processing audio chunks.
    """

    def __init__(self, client, **kwargs):
        """
        Initialize the SilenceAtEndOfChunk buffering strategy.

        Args:
            client (Client): The client instance associated with this buffering
                             strategy.
            **kwargs: Additional keyword arguments, including
                      'chunk_length_seconds' and 'chunk_offset_seconds'.
        """
        self.client = client
        self.current_chunk = bytearray()

        self.chunk_length_seconds = os.environ.get(
            "BUFFERING_CHUNK_LENGTH_SECONDS"
        )
        if not self.chunk_length_seconds:
            self.chunk_length_seconds = kwargs.get("chunk_length_seconds")
        self.chunk_length_seconds = float(self.chunk_length_seconds)

        self.chunk_offset_seconds = os.environ.get(
            "BUFFERING_CHUNK_OFFSET_SECONDS"
        )
        if not self.chunk_offset_seconds:
            self.chunk_offset_seconds = kwargs.get("chunk_offset_seconds")
        self.chunk_offset_seconds = float(self.chunk_offset_seconds)
        self.chunk_length_in_bytes = (
            self.chunk_length_seconds
            * self.client.sampling_rate
            * self.client.samples_width
        )

        self.error_if_not_realtime = os.environ.get("ERROR_IF_NOT_REALTIME")
        if not self.error_if_not_realtime:
            self.error_if_not_realtime = kwargs.get(
                "error_if_not_realtime", False
            )

    def process_audio(self, websocket, vad_pipeline, asr_pipeline):
        """
        Process audio chunks by checking their length and scheduling
        asynchronous processing.

        This method checks if the length of the audio buffer exceeds the chunk
        length and, if so, it schedules asynchronous processing of the audio.

        Args:
            websocket: The WebSocket connection for sending transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        if len(self.client.buffer) < self.chunk_length_in_bytes:
            return

        self.client.scratch_buffer += self.client.buffer
        self.client.buffer.clear()

        if len(self.current_chunk) > 0:
            print(f"Still processing {len(self.current_chunk)}, now waiting for {len(self.client.scratch_buffer)}")
            return


        self.current_chunk += self.client.scratch_buffer
        self.client.scratch_buffer.clear()
        # Schedule the processing in a separate task
        asyncio.create_task(
            self.process_audio_async(websocket, vad_pipeline, asr_pipeline)
        )

    async def process_audio_async(self, websocket, vad_pipeline, asr_pipeline):
        """
        Asynchronously process audio for activity detection and transcription.

        This method performs heavy processing, including voice activity
        detection and transcription of the audio data. It sends the
        transcription results through the WebSocket connection.

        Args:
            websocket (Websocket): The WebSocket connection for sending
                                   transcriptions.
            vad_pipeline: The voice activity detection pipeline.
            asr_pipeline: The automatic speech recognition pipeline.
        """
        start = time.time()
        vad_results = await vad_pipeline.detect_activity(self.current_chunk)

        if len(vad_results) == 0:
            self.current_chunk.clear()
            return

        last_segment_should_end_before = (
            len(self.current_chunk)
            / (self.client.sampling_rate * self.client.samples_width)
        ) - self.chunk_offset_seconds
        if vad_results[-1]["end"] < last_segment_should_end_before:
            transcription = await asr_pipeline.transcribe(self.current_chunk)
            self.current_chunk.clear()
            if transcription["text"] != "":
                end = time.time()
                transcription["processing_time"] = end - start
                json_transcription = json.dumps(transcription)
                print(f"transcribed {transcription["text"]} words in {transcription["processing_time"]} seconds")
                await websocket.send(json_transcription)
        else:
            self.current_chunk += self.client.scratch_buffer
            self.client.scratch_buffer.clear()
            self.client.scratch_buffer += self.current_chunk
            print(f"Still talking, now at {len(self.client.scratch_buffer)}")
            self.current_chunk.clear()
