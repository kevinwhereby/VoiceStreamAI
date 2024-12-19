class ASRInterface:
    async def transcribe(self):
        """
        Transcribe the given audio data.

        :param buffer: The audio buffer
        :return: The transcription structure, see for example the
                 faster_whisper_asr.py file.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
