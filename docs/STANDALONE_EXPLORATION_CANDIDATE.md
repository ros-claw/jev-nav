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

`standalone/depth_probe.py` is a separate point-projection diagnostic. It
requires an explicit rectification assumption and ground-height hypothesis,
for example `python2 standalone/depth_probe.py --assume-rectified
--ground-z=-0.1`. The result is never an authorization to move. With that
unverified hypothesis and the published TF, the observed 0.47 m optical-depth
minimum corresponded to a likely floor point around 0.58 m ahead of the base;
the full-image minimum alone would have mistaken it for a forward obstacle.

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

For a stationary on-device check on 2026-11-19, a separately built candidate
base driver replaced the running base process without overwriting its binary.
The relay sent 40 zero heartbeats; the driver reported 48 zero writes and fresh
mode 0, and odometry remained stationary. No nonzero motion command was sent.
That checks a zero-only handshake, not physical braking or failure behavior.
The candidate runs from a temporary build path and has no restart supervisor;
the original launch will use the original driver after reboot.
