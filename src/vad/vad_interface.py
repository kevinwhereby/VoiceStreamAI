class VADInterface:
    """
    Interface for voice activity detection (VAD) systems.
    """

    async def detect_activity(self, buffer):
        """
        Detects voice activity in the given audio data.

        Args:
            buffer: The audio buffer

        Returns:
            List: VAD result, a list of objects containing "start", "end",
                  "confidence".
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses."
        )
