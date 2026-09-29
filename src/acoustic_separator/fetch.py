"""Optional URL input (YouTube etc.), kept separate from the separation core.

Downloads the best available audio-only stream without re-encoding. If the download is
not possible (terms of use, DRM, network policy, bot checks) pass a local file instead.
"""
from __future__ import annotations

import re
from pathlib import Path

URL_RE = re.compile(r"^https?://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


def download_audio(url: str, out_dir: str | Path) -> Path:
    try:
        import yt_dlp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("URL input requires yt-dlp (pip install yt-dlp)") from e
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        # prefer lossless/highest-bitrate audio-only stream, no transcoding
        "format": "bestaudio[acodec=opus]/bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download {url!r} ({e}).\n"
            "URL download can fail because of terms of use, DRM, region locks or network "
            "policy. Obtain the audio legally and pass the local file with --input instead."
        ) from e
    return path
