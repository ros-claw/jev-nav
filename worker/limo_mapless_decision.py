#!/usr/bin/env python3
"""One asynchronous Decisions request for the daemon-owned mapless worker."""

import base64
import json
import math
import os
import re
import sys
import time
import urllib.request

MAX_IMAGE_BYTES = 2_000_000
MAX_AGE_SEC = 8.0
CHOICES = ("forward", "left", "right", "stop")


def _transport(payload, key, timeout):
    request = urllib.request.Request(
        "https://api.openai.com/v1/decisions",
        data=json.dumps(payload, allow_nan=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read(1_000_000))


def decide(observation, key, transport=_transport, clock=time.monotonic):
    """Return a timestamped direction from a single coherent observation."""
    rejected = {"choice": "stop", "ok": False}
    try:
        stamp = observation["captured_at"]
        image = observation["image_b64"]
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (int, float))
            or not math.isfinite(stamp)
            or not isinstance(image, str)
            or not isinstance(observation["laser"], dict)
            or not isinstance(observation["base"], dict)
            or not isinstance(key, str)
            or not key
        ):
            return rejected
        binary = base64.b64decode(image, validate=True)
        if not binary.startswith(b"\xff\xd8") or len(binary) > MAX_IMAGE_BYTES:
            return rejected
        started = clock()
        if not stamp <= started < stamp + MAX_AGE_SEC:
            return rejected
        payload = {
            "model": "gpt-6-luna",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "A differential-drive robot is exploring indoors. Choose a promising direction using the camera and current sensor data.",
                        },
                        {"type": "input_image", "image_url": "data:image/jpeg;base64," + image},
                        {
                            "type": "input_json",
                            "data": {"laser": observation["laser"], "base": observation["base"]},
                        },
                    ],
                }
            ],
            "questions": [
                {
                    "type": "choice",
                    "name": "next_action",
                    "instructions": (
                        "Select the next exploration direction, preferring progress into visible "
                        "open space when supported. The lidar may cover only the front: forward_view "
                        "groups ray directions about the laser, and reports nearest hit distance and "
                        "bearing from the robot base. max_unobserved_angle_rad includes sector edges; "
                        "max_invalid_return_run_rad describes invalid rays within the scan only. "
                        "base_hits groups all measured points by bearing from the robot base, "
                        "which can differ from the ray sector because of sensor position. "
                        "These statistics do not prove clearance. If rear_valid_rays "
                        "is zero the rear is unknown. Missing returns and nulls are unknown. "
                        "The local controller checks obstacles independently."
                    ),
                    "choices": [
                        {"value": "forward", "description": "Continue exploring ahead"},
                        {"value": "left", "description": "Explore to the left"},
                        {"value": "right", "description": "Explore to the right"},
                        {"value": "stop", "description": "Wait"},
                    ],
                }
            ],
        }
        response = transport(payload, key, 5.0)
        finished = clock()
        if not started <= finished < stamp + MAX_AGE_SEC:
            return rejected
        answers = response["answers"]
        if (
            len(answers) != 1
            or answers[0].get("name") != "next_action"
            or answers[0].get("type") != "choice"
        ):
            return rejected
        choice = answers[0].get("choice")
        if choice not in CHOICES:
            return rejected
        return {"choice": choice, "captured_at": stamp, "ok": True}
    except Exception:  # noqa: BLE001 - model/network failures return an unavailable result
        return rejected


def _key_from_config():
    path = os.environ.get("ROSCLAW_LIMO_DECISIONS_KEY_FILE")
    if not path:
        raise ValueError("API key file is not configured")
    with open(path, "r") as source:
        content = source.read(16384)
    keys = re.findall(r"\bsk-[A-Za-z0-9_-]{16,}", content)
    if len(keys) != 1:
        raise ValueError("API key file must contain one key")
    return keys[0]


def main():
    try:
        raw = sys.stdin.buffer.read(MAX_IMAGE_BYTES * 2 + 8192)
        observation = json.loads(raw)
        result = decide(observation, _key_from_config())
    except Exception:  # noqa: BLE001 - do not emit credentials or error bodies
        result = {"choice": "stop", "ok": False}
    print(json.dumps(result, allow_nan=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
