import numpy as np
import torch
from kokoro import KPipeline
from typing import List

from .base import TTSEngine


class KokoroEngine(TTSEngine):
    """TTS engine wrapping the Kokoro KPipeline model.

    Uses the Kokoro-82M model for lightweight, fast text-to-speech synthesis.
    Runs comfortably on CPU or CUDA.

    Attributes:
        _pipeline: The KPipeline TTS model.
        _voice: The voice identifier (e.g. 'af_heart').
        _speed: The playback speed multiplier.
        _sample_rate: The output sample rate in Hz.
    """

    def __init__(self,
                 voice: str = 'af_heart',
                 speed: float = 1.0,
                 pipeline: KPipeline | None = None) -> None:
        """Initialize the Kokoro engine.

        Args:
            voice: The voice identifier to use. Defaults to 'af_heart'.
            speed: The playback speed multiplier. Defaults to 1.0.
            pipeline: An optional pre-constructed KPipeline instance.
                      If None, a new pipeline is created automatically,
                      using CUDA if available.
        """
        if pipeline is None:
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
            pipeline = KPipeline(lang_code='a', device=device)
        self._pipeline: KPipeline = pipeline
        self._voice: str = voice
        self._speed: float = speed
        self._sample_rate: int = 24000

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from text using Kokoro.

        The text is split by the pipeline into segments. All segments
        are concatenated into a single audio array.

        Args:
            text: The text to synthesize into speech.

        Returns:
            A numpy array containing the generated audio samples.
        """
        audio_segments: List[np.ndarray] = []
        for i, (gs, ps, audio) in enumerate(
                self._pipeline(text, voice=self._voice, speed=self._speed, split_pattern=r'\n+')):
            print(f"  Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
            audio_segments.append(audio)
        return np.concatenate(audio_segments)
