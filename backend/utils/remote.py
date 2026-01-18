"""
Remote inference helpers for Nano Banana and Veo integrations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import base64
import time
import httpx


def encode_base64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def decode_base64_data(data: str) -> bytes:
    if data.startswith("data:"):
        parts = data.split(",", 1)
        if len(parts) == 2:
            data = parts[1]
    return base64.b64decode(data)


def _extract_asset_from_json(data: Dict[str, Any], kind: str) -> Tuple[Optional[bytes], Optional[str]]:
    if not isinstance(data, dict):
        return None, None

    result = data.get("result")
    if isinstance(result, dict):
        asset = _extract_asset_from_json(result, kind)
        if asset != (None, None):
            return asset

    if kind == "image":
        b64_keys = [
            "image_b64",
            "image_base64",
            "image",
            "preview_image_b64",
            "preview_image_base64",
            "output_image",
            "result_image",
        ]
        url_keys = [
            "image_url",
            "preview_image_url",
            "output_image_url",
        ]
    else:
        b64_keys = [
            "video_b64",
            "video_base64",
            "video",
            "output_video",
            "veo_video",
        ]
        url_keys = [
            "video_url",
            "output_video_url",
            "veo_video_url",
        ]

    for key in b64_keys:
        if key in data and data[key]:
            return decode_base64_data(str(data[key])), None

    for key in url_keys:
        if key in data and data[key]:
            return None, str(data[key])

    return None, None


def _poll_status(
    client: httpx.Client,
    status_url: str,
    kind: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> Tuple[Optional[bytes], Optional[str]]:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        response = client.get(status_url)
        response.raise_for_status()

        if response.headers.get("content-type", "").startswith("application/json"):
            data = response.json()
            status = str(data.get("status", "")).lower()
            if status in {"completed", "succeeded", "success", "done"}:
                asset = _extract_asset_from_json(data, kind)
                if asset != (None, None):
                    return asset
            if status in {"failed", "error", "cancelled"}:
                raise RuntimeError(data.get("error") or data.get("message") or "Remote generation failed.")
        else:
            return response.content, None

        time.sleep(poll_interval_sec)

    raise TimeoutError("Remote generation timed out while polling status.")


def request_remote_asset(
    url: str,
    payload: Dict[str, Any],
    kind: str,
    api_key: Optional[str] = None,
    timeout_sec: float = 600.0,
    poll_timeout_sec: float = 1800.0,
    poll_interval_sec: float = 5.0,
) -> Tuple[Optional[bytes], Optional[str]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(timeout=timeout_sec) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if content_type.startswith("application/json"):
            data = response.json()

            asset = _extract_asset_from_json(data, kind)
            if asset != (None, None):
                return asset

            status_url = data.get("status_url") or data.get("poll_url")
            if status_url:
                return _poll_status(
                    client,
                    status_url,
                    kind,
                    timeout_sec=poll_timeout_sec,
                    poll_interval_sec=poll_interval_sec,
                )

            raise RuntimeError("Remote generation response missing asset payload.")

        return response.content, None


def download_to_path(url: str, output_path: str, timeout_sec: float = 300.0) -> str:
    with httpx.Client(timeout=timeout_sec) as client:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
    return output_path
