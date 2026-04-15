"""Minimal HTTP client for the unified VLA ↔ benchmark protocol.

Matches `VLA_COMMUNICATION_PROTOCOL.md` so any server (pi0.5, X-VLA, DreamVLA,
...) that implements `/health`, `/reset`, `/act` can be driven from here without
modification.

Intentionally stays in stdlib + numpy + Pillow + requests so that it runs inside
the Libero-pro container (Python 3.8, robosuite 1.4.0) with no new deps.
"""

from __future__ import annotations

import base64
import io
import time
from typing import Dict, Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image


def encode_image(img: np.ndarray) -> str:
    """HxWx3 uint8 numpy (RGB) → base64 PNG string."""
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class VLAClient:
    def __init__(self, url: str, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def health_check(self) -> Optional[dict]:
        try:
            r = requests.get(f"{self.url}/health", timeout=5.0)
            if r.status_code != 200:
                return None
            return r.json()
        except requests.exceptions.RequestException:
            return None

    def wait_until_ready(self, max_wait: float = 180.0, poll_interval: float = 3.0) -> dict:
        t0 = time.time()
        last = None
        while time.time() - t0 < max_wait:
            info = self.health_check()
            if info is not None and info.get("status") == "ok":
                return info
            last = info
            time.sleep(poll_interval)
        raise TimeoutError(
            f"VLA server at {self.url} not ready after {max_wait}s. Last health: {last}"
        )

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        r = requests.post(f"{self.url}/reset", json={}, timeout=self.timeout)
        r.raise_for_status()

    def predict(
        self,
        images: Dict[str, np.ndarray],
        states: Optional[Dict[str, np.ndarray]],
        instruction: str,
    ) -> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], float]:
        """Send observation, receive action.

        Args:
            images: {camera_name: HxWx3 uint8 RGB}. Keys become
                `observation.images.{camera_name}`.
            states: {state_field: 1-D ndarray}. Keys are used verbatim, so pass
                them already prefixed with `observation.state.`.
            instruction: natural-language task description.

        Returns:
            (actions, latency_ms) where `actions` is either a dict of sub-keys
            (e.g. `{"action.eef_pos": ndarray[N,3], ...}`) or a 2-D ndarray if
            the server returned the legacy flat `action` key.
        """
        payload: dict = {"task": instruction}
        for cam, img in images.items():
            payload[f"observation.images.{cam}"] = encode_image(img)
        if states is not None:
            for k, v in states.items():
                arr = np.asarray(v).reshape(-1).astype(np.float32)
                payload[k] = arr.tolist()

        t0 = time.time()
        r = requests.post(f"{self.url}/act", json=payload, timeout=self.timeout)
        r.raise_for_status()
        latency_ms = (time.time() - t0) * 1000.0
        result = r.json()

        sub_keys = [k for k in result if k.startswith("action.")]
        if sub_keys:
            out = {}
            for k in sub_keys:
                arr = np.asarray(result[k], dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[np.newaxis, :]
                out[k] = arr
            return out, latency_ms

        flat = np.asarray(result["action"], dtype=np.float32)
        if flat.ndim == 1:
            flat = flat[np.newaxis, :]
        return flat, latency_ms
