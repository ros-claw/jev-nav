# Standalone LIMO exploration candidate

This is an experimental route that does not require the ROSClaw daemon. The
standalone observer connects directly to the existing ROS 1 graph, calls the
Decisions API with camera and forward-lidar evidence, and reports the local
policy's hypothetical command. **The observer does not publish `/cmd_vel`.**
The forward-only control session and the separate ROS relay are components;
they are not yet connected into a released autonomous exploration runtime.

To run the observation-only diagnostic on the LIMO, with a separately supplied
key file containing the Decisions API key:

```bash
source /home/agilex/agilex_ws/devel/setup.bash
export ROSCLAW_LIMO_DECISIONS_KEY_FILE=/path/to/private.env
python2 standalone/explore.py --seconds 12
```

The output includes the model choice, hypothetical zero or forward command,
the latest scan's raw NaN/zero/valid bin counts, and a 3-by-3 depth-validity
summary. Counts describe raw values; their meaning depends on the exact
publisher. Missing or stale measurements remain unknown. Pose continuity,
clock jumps and approximate sensor alignment are checked locally; camera
calibration, the scan's acquisition timing and complete three-dimensional
coverage have not been established. The current policy executes no turns and
its thresholds are initial hypotheses, not measured braking or clearance
guarantees. A single zero after relay motion ends that relay session; automatic
resumption requires further integration and stopped-odometry verification.
The sampled forward cross-section may use recent ray endpoints up to 0.45 s
old, with a tighter check on the latest scan. It retains recent obstacle hits,
but cannot exclude an obstacle that entered a previously sampled gap later.
The supervised single-trial adapter uses a separate, explicitly bounded
history window of up to 0.65 s and a 0.40 s acquisition span to cope with this
lidar's sparse angular bins. Before a command it independently checks the
newest scan for measured points in a padded corridor and requires that scan to
be no more than 0.30 s old. These are sampling heuristics, not a clearance
certificate; unreturned rays and small obstacles between rays remain unknown.

`standalone/depth_probe.py` is a separate point-projection diagnostic. It
requires an explicit rectification assumption and ground-height hypothesis,
for example `python2 standalone/depth_probe.py --assume-rectified
--ground-z=-0.1`. The result is never an authorization to move. With that
unverified hypothesis and the published TF, the observed 0.47 m optical-depth
minimum corresponded to a likely floor point around 0.58 m ahead of the base;
the full-image minimum alone would have mistaken it for a forward obstacle.
`ProjectedDepthGate` can use the same kind of same-frame projection in an
explicitly configured supervised trial. It rejects measured obstacle and
unclassified corridor points, while reporting unobserved space and the
unverified calibration. On this device, six diagnostic frames each classified
more than a thousand sampled ground points and no sampled obstacle points in
the configured corridor; this does not establish that the full path is clear.
`standalone/depth_visibility.py` can also project explicit body-frame sample
centers into the depth image and classify each sampled ray; it does not certify
the volume between those centers. The installed camera identifies itself as
DaBai DC1. The [manufacturer's DaBai series specification](https://developer-orbbec-oss.oss-cn-shenzhen.aliyuncs.com/kindeditor/20220805190815.pdf)
and [AgileX's DaBai documentation](https://github.com/agilexrobotics/limo_pro_doc/blob/master/Limo%20Pro%20user%20manual%28EN%29.md)
describe a 0.3–3 m operating range. The observer's depth-validity diagnostic
uses that range, but it is not a substitute for validating this device's
calibration and actual performance.

For one live diagnostic frame, using that range, a 0.02 m near-limit margin,
and the published but unverified transform, all 12 sampled centers at each of
base x = 0.2, 0.3 and 0.4 m had unknown depth visibility. Across 84 sampled
centers at x = 0.2–1.0 m, 79 were unknown and five had a depth beyond the
center. These are sampled observations, not a complete coverage measurement.

On the observed vehicle the existing X2L driver resized the SDK's point
vector in fixed-resolution mode, padding it with fake zero-angle points or
truncating real points. It also represented both unfilled angular bins and
SDK-invalid returns as zero. `patches/ydlidar-diagnostic-candidate.patch` is
an isolated candidate against the locally modified `ydlidar_ros` tree: it
retains the actual point vector and represents unfilled bins as NaN while
leaving assigned invalid readings as zero. Duplicate bins retain their nearest
valid hit; a single `LaserScan` range cannot preserve every duplicate point.
The patch does not recover missing rays or prove free space. Check it against
the exact local source before applying it and preserve local launch changes.

The candidate was tested while the vehicle was stationary on 2026-11-19. In
20 scans, there were approximately 181–198 valid points, 18–55 assigned zero
points and 206–248 unassigned bins per scan; the median scan period was about
0.113 s. The isolated node was started separately, leaving the original binary
untouched. A reboot or the original launch uses the original driver until the
candidate is deliberately installed. The depth camera's auto exposure was
enabled for this trial and improved valid depth coverage, though some image
areas remained largely invalid.

The candidate base watchdog and forward-only relay have been tested together
with a fake serial device in a network/device-isolated namespace. Killing the
relay caused the candidate driver to latch zero and reject replayed commands
in that simulation. Neither a serial write nor those tests prove that the
physical chassis has stopped. The watchdog cannot act when the driver process
or host is unavailable, and an independently verified physical stop and
exclusive command arbitration remain necessary before real autonomy.

The standalone relay also accepts `--max-motion-sec` to latch a zero command
after a bounded interval from its first nonzero frame, even if fresh heartbeats
continue. In an isolated PTY test, continuous fake motion heartbeats reached
`STOP motion_deadline` after approximately 0.305 seconds; the first subsequent
zero serial frame was observed about 0.03–0.05 seconds after the stop report.
A real zero-only handshake separately confirmed the requested `0.3` second
configuration and 20 zero heartbeats. These tests do not measure a physical
stopping time or account for host failure.

`standalone/base_sentry.py` is a PTY-only research fixture, not a hardware
watchdog. In the modeled driver parser, writing zero padding before zero-speed
frames avoided interpreting truncated bounded forward commands as new nonzero
commands. The actual chassis parser, serial buffering, device ownership and
an inherited serial descriptor are not established; the fixture refuses real
device paths and does not protect a real LIMO.

For a stationary on-device check on 2026-11-19, a separately built candidate
base driver replaced the running base process without overwriting its binary.
The relay sent 40 zero heartbeats; the driver reported 48 zero writes and fresh
mode 0, and odometry remained stationary. No nonzero motion command was sent.
That checks a zero-only handshake, not physical braking or failure behavior.
The candidate runs from a temporary build path and has no restart supervisor;
the original launch will use the original driver after reboot.

The subsequent candidate with serial `TIOCEXCL` was started on the stationary
robot, after checking for existing serial users; a new ordinary open returned
`EBUSY`. Its relay completed a further zero-only handshake and odometry stayed
stationary. The process and ROS caller monitors are implemented separately in
`standalone/live_monitors.py`: their evidence is a non-atomic observation, not
proof against an already open descriptor, hidden processes or privileged access.
On the tested machine `fuser` took approximately 0.20 seconds at rest and was
occasionally slower during camera processing. The supervised trial uses a
0.40-second subprocess limit and a 0.60-second serial-monitor age limit; its
separate ROS watchdog-status monitor keeps the shorter freshness limit. Process
identity is checked frequently; the more expensive `fuser` scan is reused for
at most 1.2 seconds and refreshed approximately every 0.75 seconds. A new
ordinary open remains blocked by the candidate's initial `TIOCEXCL`, subject to
the previously stated limitations.
