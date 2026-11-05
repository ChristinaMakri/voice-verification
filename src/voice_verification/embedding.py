"""Speaker-embedding models."""

import os
from typing import Protocol

import numpy as np

from .scoring import l2_normalize


class Embedder(Protocol):
    model_id: str
    sample_rate: int

    def embed(self, samples: np.ndarray) -> np.ndarray:
        """Return an L2-normalised speaker embedding for mono float32 audio."""
        ...


class EcapaEmbedder:
    """ECAPA-TDNN trained on VoxCeleb (SpeechBrain), 192-dim embeddings at 16 kHz.

    Trained on wideband web video, not telephone speech: expect a lower score
    margin on 8 kHz calls and calibrate accordingly.
    """

    model_id = "speechbrain/spkrec-ecapa-voxceleb"
    sample_rate = 16000

    def __init__(self, cache_dir: str = "./.models", device: str = "cpu"):
        try:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise RuntimeError(
                "the ECAPA model needs the 'ecapa' extra: pip install '.[ecapa]'"
            ) from exc
        self._torch = torch
        self._model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=os.path.join(cache_dir, "spkrec-ecapa-voxceleb"),
            run_opts={"device": device},
        )

    def embed(self, samples: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            wav = self._torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
            emb = self._model.encode_batch(wav).squeeze().cpu().numpy()
        return l2_normalize(emb.astype(np.float32))
