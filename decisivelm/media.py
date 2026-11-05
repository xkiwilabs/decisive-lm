"""Images and audio in the shared prefix (docs/guide/06 §6.10): loading, encoding and their placeholder tokens.

An image enters the prefix as `<|image>`, one `<|image|>` soft token per pooled patch group, then `<image|>`, which is
exactly what Gemma 4's processor writes for a lone image placeholder. Its pixels travel beside the token ids in the
packed tree, and the backbone scatters the vision tower's output into the soft-token slots in order. Every
image of a request is encoded at one fixed soft-token budget (Gemma 4: 70, 140, 280, 560 or 1120), recorded per item,
so teacher and student see the same tokens.

An audio clip enters the same way, as `<|audio>`, one `<|audio|>` soft token per 40 ms, then `<audio|>`: its waveform
is resampled to the feature extractor's rate (16 kHz), cut at MAX_AUDIO_SECONDS, and turned into log-mel features by
Gemma 4's own feature extractor. The soft-token count follows the processor's arithmetic (two stride-2 convolutions
over the feature mask), so it matches what the audio tower returns. Decoding needs the `audio` extra (soundfile,
scipy).
"""
import base64
import binascii
import io
import pathlib
from collections.abc import Callable
from dataclasses import dataclass

import torch
from PIL import Image

from decisivelm.api_types import Media

DEFAULT_BUDGET = 280
MAX_AUDIO_SECONDS = 30   # Gemma 4's feature extractor and processor cap (480,000 samples, 750 soft tokens)
SAMPLE_RATES = {8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 88_200, 96_000}


@dataclass
class ImageTokens:
    boi: int
    image: int
    eoi: int

    @classmethod
    def from_config(cls, config) -> "ImageTokens | None":
        """From a Gemma4Config (Gemma4Model.config); None for a text-only backbone."""
        if getattr(config, "vision_config", None) is None:
            return None
        return cls(config.boi_token_id, config.image_token_id, config.eoi_token_id)


@dataclass
class EncodedImage:
    token_ids: list[int]                       # boi, n soft tokens, eoi
    pixel_values: torch.Tensor | None          # [P, patch_dim]; None once dropped (see `drop_pixels`)
    position_ids: torch.Tensor | None          # [P, 2] patch (x, y); padding patches (-1, -1)
    budget: int
    reload: Callable[[], "EncodedImage"] | None = None

    @property
    def n_soft(self) -> int:
        return len(self.token_ids) - 2

    def pixels(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.pixel_values is None:
            again = self.reload()
            if again.token_ids != self.token_ids:
                raise ValueError("an image re-encoded to different tokens (its file changed?)")
            return again.pixel_values, again.position_ids
        return self.pixel_values, self.position_ids

    def drop_pixels(self):
        """Keep the tokens, free the pixels (~8 MB at 280 soft tokens); `pixels()` re-encodes from the source.
        Training keeps every row's tree, so it holds images this way and loads pixels per micro-batch."""
        if self.reload is not None:
            self.pixel_values = self.position_ids = None


def _decode(data: str, kind: str) -> io.BytesIO:
    """Base64 media bytes; a `data:<type>;base64,` prefix is accepted. Not base64 is a ValueError (a 400 in server
    mode), never garbage passed on to a decoder."""
    if data.startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        return io.BytesIO(base64.b64decode(data, validate=True))
    except binascii.Error as e:
        raise ValueError(f"{kind} data is not valid base64 ({e})") from e


def load_image(m: Media, root: pathlib.Path | None = None) -> Image.Image:
    """A Media item as an RGB image. `path` is resolved against `root` when relative. URLs are refused here: server
    mode fetches only allow-listed URLs (docs/guide/04 §4.5), and that belongs in the server, not in the model."""
    if m.type != "image":
        raise NotImplementedError(f"{m.type} media are not supported yet")
    if m.path is not None:
        p = pathlib.Path(m.path)
        img = Image.open(p if p.is_absolute() or root is None else root / p)
    elif m.data is not None:
        try:
            img = Image.open(_decode(m.data, "image"))
            img.load()
        except (OSError, Image.DecompressionBombError) as e:
            raise ValueError(f"image data is not a readable image ({e})") from e
    else:
        raise ValueError("image URLs must be fetched by the caller (allow-listed in server mode)")
    return img.convert("RGB")


def encode_image(image_processor, image: Image.Image, tokens: ImageTokens, budget: int = DEFAULT_BUDGET) -> EncodedImage:
    out = image_processor(images=[image], max_soft_tokens=budget, return_tensors="pt")
    n = int(out["num_soft_tokens_per_image"][0])
    return EncodedImage(token_ids=[tokens.boi] + [tokens.image] * n + [tokens.eoi],
                        pixel_values=out["pixel_values"][0], position_ids=out["image_position_ids"][0], budget=budget)


def stack(images: list[EncodedImage]) -> tuple[torch.Tensor, torch.Tensor]:
    """Images of one packed batch as the backbone's (pixel_values [n, P, d], image_position_ids [n, P, 2]); images of
    different budgets are padded to the longest with zero pixels at (-1, -1), which the vision tower drops."""
    loaded = [im.pixels() for im in images]
    P = max(len(x) for x, _ in loaded)
    d = loaded[0][0].shape[-1]
    pix = torch.zeros(len(images), P, d, dtype=loaded[0][0].dtype)
    pos = torch.full((len(images), P, 2), -1, dtype=loaded[0][1].dtype)
    for i, (x, xy) in enumerate(loaded):
        pix[i, :len(x)] = x
        pos[i, :len(xy)] = xy
    return pix, pos


@dataclass
class AudioTokens:
    boa: int
    audio: int
    eoa: int

    @classmethod
    def from_config(cls, config) -> "AudioTokens | None":
        """From a Gemma4Config; None for a backbone without an audio tower (26B A4B)."""
        if getattr(config, "audio_config", None) is None:
            return None
        return cls(config.boa_token_id, config.audio_token_id, config.eoa_token_index)


@dataclass
class EncodedAudio:
    token_ids: list[int]                       # boa, n soft tokens, eoa
    features: torch.Tensor | None              # [T, n_mels] log-mel frames; None once dropped
    mask: torch.Tensor | None                  # [T] True for real frames
    reload: Callable[[], "EncodedAudio"] | None = None

    @property
    def n_soft(self) -> int:
        return len(self.token_ids) - 2

    def frames(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.features is None:
            again = self.reload()
            if again.token_ids != self.token_ids:
                raise ValueError("an audio clip re-encoded to different tokens (its file changed?)")
            return again.features, again.mask
        return self.features, self.mask

    def drop_features(self):
        """Keep the tokens, free the features; `frames()` re-encodes from the source (as EncodedImage.drop_pixels)."""
        if self.reload is not None:
            self.features = self.mask = None


def load_audio(m: Media, root: pathlib.Path | None = None, sampling_rate: int = 16_000):
    """A Media item as a mono float32 waveform at `sampling_rate`, cut at MAX_AUDIO_SECONDS. `path` is resolved
    against `root` when relative; URLs are refused, as for images."""
    import numpy as np
    try:
        import soundfile
        from scipy.signal import resample_poly
    except ImportError as e:
        raise ValueError("audio needs the `audio` extra: uv sync --extra audio (soundfile, scipy)") from e
    if m.type != "audio":
        raise ValueError(f"not an audio item: {m.type}")
    if m.path is not None:
        p = pathlib.Path(m.path)
        source = p if p.is_absolute() or root is None else root / p
    elif m.data is not None:
        source = _decode(m.data, "audio")
    else:
        raise ValueError("audio URLs must be fetched by the caller (allow-listed in server mode)")
    try:
        f = soundfile.SoundFile(source)
    except (RuntimeError, OSError) as e:   # soundfile.LibsndfileError is a RuntimeError
        raise ValueError(f"audio is not a readable sound file ({e})") from e
    with f:
        # Checked from the header before decoding: only the first MAX_AUDIO_SECONDS are read, and an unusual rate
        # (which would make the resampling ratio, and its cost, large) is refused.
        if f.samplerate not in SAMPLE_RATES or not 1 <= f.channels <= 8:
            raise ValueError(f"unsupported audio: {f.samplerate} Hz, {f.channels} channels")
        sr = f.samplerate
        wave = f.read(frames=MAX_AUDIO_SECONDS * sr, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if sr != sampling_rate:
        g = np.gcd(sr, sampling_rate)
        wave = resample_poly(wave, sampling_rate // g, sr // g).astype(np.float32)
    return wave[:MAX_AUDIO_SECONDS * sampling_rate]


def audio_soft_tokens(mask: torch.Tensor) -> int:
    """Soft tokens for a feature mask: two stride-2 convolutions (kernel 3, padding 1), as Gemma4Processor counts."""
    for _ in range(2):
        mask = mask[::2][:(len(mask) + 2 - 3) // 2 + 1]
    return int(mask.sum())


def encode_audio(feature_extractor, wave, tokens: AudioTokens) -> EncodedAudio:
    out = feature_extractor([wave], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
    feats, mask = out["input_features"][0], out["input_features_mask"][0].bool()
    n = audio_soft_tokens(mask)
    if n == 0:
        raise ValueError("audio clip too short to encode")
    return EncodedAudio(token_ids=[tokens.boa] + [tokens.audio] * n + [tokens.eoa], features=feats, mask=mask)


def stack_audio(clips: list[EncodedAudio]) -> tuple[torch.Tensor, torch.Tensor]:
    """Clips of one packed batch as the backbone's (input_features [n, T, n_mels], input_features_mask [n, T]),
    padded to the longest with zero frames that the mask marks as padding."""
    loaded = [c.frames() for c in clips]
    T, d = max(len(f) for f, _ in loaded), loaded[0][0].shape[-1]
    feats = torch.zeros(len(clips), T, d, dtype=loaded[0][0].dtype)
    mask = torch.zeros(len(clips), T, dtype=torch.bool)
    for i, (f, m) in enumerate(loaded):
        feats[i, :len(f)], mask[i, :len(m)] = f, m
    return feats, mask
