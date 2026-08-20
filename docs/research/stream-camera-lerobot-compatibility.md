# `camera-stream` and LeRobot camera compatibility

## Decision

Implement a new LeRobot `Camera` backend as a thin adapter around the public
`camera_stream.StreamClient` API. Do not use LeRobot's built-in `ZMQCamera`:
it expects a one-part JSON message containing Base64 JPEG images, whereas
`camera-stream` publishes three-part multipart messages. `camera-stream` 0.3.3
now supports Python `>=3.10`, so it is compatible with this repository's
Python 3.12+ requirement and should own the transport, wire validation, and
JPEG decoding. [1][2][3][4]

## LeRobot contract

This repository pins `lerobot==0.6.0`. A backend configuration should be a
dataclass subclass of `lerobot.cameras.CameraConfig`, registered under a new
choice name (for example, `"stream"`). The base fields are `fps`, `width`, and
`height`; registration makes its `type` available to LeRobot configuration
parsing. `make_cameras_from_configs()` constructs known built-ins directly and
otherwise calls LeRobot's registered-device factory, so the config class and
camera class must be importable before factory construction. [3][5]

The camera class must subclass `lerobot.cameras.Camera` and implement:

- `is_connected`: true only while the SUB socket and its receiver are usable.
- `find_cameras()`: there is no discovery endpoint in `camera-stream`; raise
  `NotImplementedError` (or document a status-based future extension).
- `connect(warmup=True)`: create the subscriber, subscribe to the selected
  `<camera>/color` topic, start receiving, determine the actual dimensions
  from a frame when absent, and, when requested, wait for the first valid
  frame.
- `read()`: wait for the next *unconsumed* frame. `async_read(timeout_ms=200)`
  has the same freshness semantics but a caller-controlled timeout.
- `read_latest(max_age_ms=...)`: return the retained latest frame without
  consuming it; raise `TimeoutError` when it is too old.
- `disconnect()`: stop the receiver before closing its socket/context and
  clear frame state. The LeRobot base also supplies context-manager cleanup.

These semantics align with the service's explicit latest-frame-wins design:
slow consumers drop old frames instead of accumulating latency. [3][6]

## Proposed public configuration

The minimal stable adapter configuration is:

```python
@CameraConfig.register_subclass("stream")
@dataclass
class StreamCameraConfig(CameraConfig):
    endpoint: str                       # e.g. "tcp://192.168.5.24:5555"
    camera_name: str                    # e.g. "hand_camera"
    color_mode: ColorMode = ColorMode.RGB
    timeout_ms: int = 5_000             # startup / receive timeout
    warmup_s: float = 5.0
```

Validate that `endpoint` is non-empty and uses `tcp://`, `camera_name` is
non-empty, `timeout_ms > 0`, and `warmup_s >= 0`. Keep `fps`, `width`, and
`height` optional inherited fields. The stream is authoritative for actual
dimensions; reject a received frame when configured dimensions disagree rather
than silently reshaping it. The service's camera names are restricted to
`[A-Za-z0-9_.-]+`, so joining the public topic as
`f"{camera_name}/color"` is unambiguous. [3][6]

`camera-stream` frames are BGR8. LeRobot's `ColorMode` defaults to RGB, and
the native OpenCV LeRobot backend explicitly converts BGR to RGB for that
mode. The stream backend should apply the same conversion at its camera
boundary and leave BGR unchanged when configured. [3][7]

## `camera-stream` client contract (0.3.3)

The backend calls `StreamClient(endpoint).subscribe("<camera-name>/color")`.
The client subscribes with ZeroMQ `SUBSCRIBE` to the topic and receives each
image as a three-part message:

```text
[topic UTF-8] [header JSON UTF-8] [JPEG or raw BGR bytes]
```

Validate before decoding:

| Header / payload requirement | Required value or check |
| --- | --- |
| `schema_version` | integer `1` |
| routing | topic equals `<camera>/color`; `camera` is non-empty and `stream == "color"` |
| image description | `pixel_format == "bgr8"`; `codec` is `"jpeg"` or `"raw_bgr8"` |
| timestamps | `timestamp_source == "host"`; integer `captured_monotonic_ns` and `captured_utc_ns` |
| dimensions and ordering | positive integer `width`, `height`, and integer `sequence` |
| payload integrity | integer `payload_size == len(payload)` |
| JPEG decode | `cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)` must succeed |
| raw decode | require `payload_size == width * height * 3`, then reshape `uint8` data to `(height, width, 3)` |

The service sets low-capacity queues throughout; frame sequence gaps are
expected under load, reset when a worker restarts, and must not be treated as
a reconnect failure. Its timestamps are generated on the server host after
`driver.read()` returns. `captured_utc_ns` is suitable for logging only when
host clocks are synchronized; `captured_monotonic_ns` is meaningful only on
that server host. [6][8]

The service also publishes two-part status messages on `status/snapshot` and
`status/camera/<camera>`. They are useful for diagnostics but are not required
to read a camera. With server idle policy enabled, subscribing to
`<camera>/color` is what wakes or keeps a camera active; subscribing only to
`status/` does not. [6]

## Runtime and test implications

- Depend on the verified `camera-stream` client version. It already owns the
  ZeroMQ context, `RCVHWM=1`, `LINGER=0`, receiver thread, protocol validation,
  JPEG decoding, and capacity-one latest-frame behavior. The LeRobot adapter
  should only map `StreamClient` reads to LeRobot's read methods, verify output
  dimensions, and convert BGR to RGB when requested. [4][8]
- Tests should run a local `zmq.PUB` fixture and cover topic filtering, both
  codecs, malformed header/payload rejection, BGR-to-RGB conversion,
  `async_read` timeout and consumption, `read_latest` staleness, clean
  disconnect, and dimension mismatch. Use a short publisher/subscriber
  synchronization delay because PUB/SUB does not guarantee messages sent
  before a subscription is established. [6][8]

## Sources

1. Hugging Face, [LeRobot camera documentation](https://huggingface.co/docs/lerobot/main/en/cameras), accessed 2026-08-20.
2. This repository, [Python requirement and LeRobot pin](../../pyproject.toml), accessed 2026-08-20.
3. Hugging Face LeRobot [`Camera`, `CameraConfig`, and camera factory at v0.6.0](https://github.com/huggingface/lerobot/tree/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/cameras), accessed 2026-08-20.
4. `camera-stream`, [project metadata for 0.3.3](https://pypi.org/project/camera-stream/0.3.3/), accessed 2026-08-20.
5. Hugging Face LeRobot, [`make_cameras_from_configs`](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/cameras/utils.py), accessed 2026-08-20.
6. `camera-stream`, [client API and wire-protocol reference](https://github.com/sorelferris/camera-stream/blob/main/README.md), accessed 2026-08-20.
7. Hugging Face LeRobot, [OpenCV color-mode conversion at v0.6.0](https://github.com/huggingface/lerobot/blob/30da8e687a6dfc617fcd94afc367ac7071c376ce/src/lerobot/cameras/opencv/camera_opencv.py), accessed 2026-08-20.
8. `camera-stream`, [validated client protocol and latest-frame client implementation](https://github.com/sorelferris/camera-stream/tree/main/src/camera_stream/client), accessed 2026-08-20.
