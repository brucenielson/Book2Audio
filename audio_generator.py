import numpy as np
import soundfile as sf

from engines import TTSEngine


class AudioGenerator:
    """Handles audio generation and file saving using a pluggable TTS engine.

    Wraps any TTSEngine implementation and provides methods to generate audio
    from text, save audio to disk, or do both in one step.

    Attributes:
        _engine: The TTSEngine used for speech synthesis.
    """

    def __init__(self, engine: TTSEngine) -> None:
        """Initialize the AudioGenerator.

        Args:
            engine: A TTSEngine instance to use for speech synthesis.
        """
        self._engine: TTSEngine = engine

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from a text string.

        Args:
            text: The text to synthesize into speech.

        Returns:
            A numpy array containing the generated audio samples.
        """
        return self._engine.generate(text)

    def save(self, audio: np.ndarray, output_file: str) -> None:
        """Save a numpy audio array to a WAV file.

        Args:
            audio: A numpy array of audio samples to save.
            output_file: The path to the output WAV file.
        """
        sf.write(output_file, audio, self._engine.sample_rate)
        print(f"Audio saved to {output_file}")

    def generate_and_save(self, text: str, output_file: str) -> None:
        """Generate audio from text and save it directly to a WAV file.

        Args:
            text: The text to synthesize into speech.
            output_file: The path to the output WAV file.
        """
        self.save(self.generate(text), output_file)
