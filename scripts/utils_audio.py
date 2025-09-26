import soundfile as sf
import numpy as np
import librosa

TARGET_SR = 16000
WIN_SEC = 2.0
WIN_SAMPLES = int(TARGET_SR * WIN_SEC)

def load_wav(path, target_sr=TARGET_SR):
    wav, sr = sf.read(path, always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1) 
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    maxv = np.max(np.abs(wav)) + 1e-9
    wav = wav / maxv
    return wav.astype(np.float32), target_sr

def crop_or_pad_2s(wav, win_samples=WIN_SAMPLES, rng=None):
    n = len(wav)
    if n >= win_samples:
        start = (0 if rng is None else rng.integers(0, n - win_samples + 1))
        return wav[start:start+win_samples]
    else:
        pad = win_samples - n
        return np.pad(wav, (0, pad))
