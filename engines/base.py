from abc import ABC, abstractmethod
import numpy as np


class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines.

    All TTS engines must implement generate() and expose a sample_rate property.
    This allows AudioGenerator and BookToAudio to work with any TTS backend
    without knowing its implementation details.
    """

    @abstractmethod
    def generate(self, text: str) -> np.ndarray:
        """Generate audio from a text string.

        Args:
            text: The text to synthesize into speech.

        Returns:
            A numpy array containing the generated audio samples.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The sample rate in Hz of the audio produced by this engine."""
        ...
