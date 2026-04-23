import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

from .base import TTSEngine
from utils.logging_utils import vprint

# Map short model size names to full Hugging Face model identifiers.
QWEN_MODEL_SIZES = {
    '0.6b': 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
    '1.7b': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
}

# Default speakers available in CustomVoice models.
QWEN_SPEAKERS = ['aiden', 'dylan', 'eric', 'ono_anna', 'ryan', 'serena', 'sohee', 'uncle_fu', 'vivian']


class QwenCustomVoiceEngine(TTSEngine):
    """TTS engine wrapping Qwen3-TTS CustomVoice models.

    Uses a predefined speaker voice with optional style instructions.
    Requires CUDA with bfloat16 support. The 0.6B model is recommended
    for GPUs with 6GB VRAM or less.

    Attributes:
        _model: The Qwen3TTSModel instance.
        _speaker: The speaker name (e.g. 'vivian').
        _language: The language code (e.g. 'English', 'Auto').
        _instruct: Optional style instruction (e.g. 'speak calmly').
        _sample_rate: The output sample rate in Hz (set after first generation).
    """

    def __init__(self,
                 speaker: str = 'vivian',
                 language: str = 'Auto',
                 instruct: str | None = None,
                 model_size: str = '0.6b',
                 model: Qwen3TTSModel | None = None,
                 verbose: bool = False) -> None:
        """Initialize the Qwen CustomVoice engine.

        Args:
            speaker: The speaker name. Defaults to 'vivian'.
                     Use QWEN_SPEAKERS to see available options.
            language: The language for synthesis. Defaults to 'Auto'
                      for automatic detection. Can also be 'English',
                      'Chinese', 'Japanese', etc.
            instruct: Optional natural-language style instruction,
                      e.g. 'speak with excitement'. None means no
                      instruction (neutral delivery).
            model_size: The model size to use: '0.6b' or '1.7b'.
                        Defaults to '0.6b'.
            model: An optional pre-loaded Qwen3TTSModel instance.
                   If None, the model is loaded based on model_size.
            verbose: If True, prints the model ID and device during loading.
                     Defaults to False.
        """
        if model is None:
            model_id: str = QWEN_MODEL_SIZES.get(model_size.lower(), QWEN_MODEL_SIZES['0.6b'])
            # Determine attention implementation: use flash_attention_2 if available, fall back to sdpa.
            attn_impl: str = 'sdpa'
            try:
                import flash_attn  # noqa: F401
                attn_impl = 'flash_attention_2'
            except ImportError:
                pass
            device: str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            vprint(verbose, f"Loading Qwen3-TTS model: {model_id} (device: {device}, attention: {attn_impl})")
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                dtype=torch.bfloat16,
                attn_implementation=attn_impl,
            )
        self._model: Qwen3TTSModel = model
        self._speaker: str = speaker
        self._language: str = language
        self._instruct: str | None = instruct
        self._verbose: bool = verbose
        self._sample_rate: int = 24000  # Will be confirmed on first generate call.

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate(self, text: str) -> np.ndarray:
        """Generate audio from text using Qwen3-TTS CustomVoice.

        Args:
            text: The text to synthesize into speech.

        Returns:
            A numpy array containing the generated audio samples.
        """
        kwargs = {
            'text': text,
            'language': self._language,
            'speaker': self._speaker,
        }
        if self._instruct is not None:
            kwargs['instruct'] = self._instruct

        wavs, sr = self._model.generate_custom_voice(**kwargs)
        self._sample_rate = sr
        return wavs[0]
