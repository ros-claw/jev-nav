# Standalone LIMO navigation

`run_navigation.py` evaluates short forward segments and shallow left and right arcs against a rolling grid in `odom`. Decisions receives the image, base state, checked starts, blocked segment reasons, and a forward LiDAR summary when the scan is synchronized with the image. The 20 Hz local controller follows the accepted segment, checks its next swept area, and can start a prepared successor without stopping. A delayed or failed request leaves the active segment in place. If sensing drops out, the controller sends zero and can resume on a fresh check of the same segment. A measured obstruction cancels that segment.

The default ROS output is an isolated `navigation_command_preview` topic. The entry point reads laser, depth, RGB and odometry. The rolling grid publishes the laser and depth acquisition stamps used for that grid. The ROS depth pipeline decimates images by two on each axis before converting them to point clouds. [ROS image decimation source](https://github.com/ros-perception/image_pipeline/blob/melodic/image_proc/src/nodelets/crop_decimate.cpp)

Build the repository version of [`limo_fresh_layer`](limo_fresh_layer) in a Catkin workspace and source that workspace and the hardware workspaces. For the tested LIMO with the X2L lidar and DaBai U3 camera, bring up the matching hardware profile once from the repository root.

```sh
roslaunch standalone/navigation_hardware.launch
```

This profile expects the [watchdog-capable LIMO driver](../docs/DRIVER_WATCHDOG_CANDIDATE.md), the X2L launch from `ydlidar_ros`, and an `astra_camera` build with the `keep_streaming` and `tf_publish_rate` arguments. The fixed sensor and optical transforms are latched on `/tf_static`; odometry remains time-varying. Depth projection can use each frame's acquisition time as soon as the corresponding odometry transform is available. Verify the mounting coordinates in the launch file before using it on another installation.

For a stationary lidar coverage check, set `laser_angle_min:=-180 laser_angle_max:=180` on `roslaunch`. The installed LIMO covered the front while the rear bearings needed for both arcs had no valid returns. The launch also accepts `laser_x`, `laser_y`, `laser_z`, and `laser_yaw` for a measured lidar mounting position and orientation.

Run a recorded live preview in another shell with the same ROS environment.

```sh
/usr/bin/python2 standalone/run_navigation.py --decisions --seconds 48 --timeline /tmp/jev-navigation.jsonl
```

The timeline path must be new. Omitting `--seconds` runs the preview until shutdown. The test `test_runtime_tracks_for_thirty_seconds_through_multiple_delayed_model_choices` runs 31 simulated seconds of odometry feedback through three decisions and tests stopping on missing input.

`--drive-forward` enables supervised straight operation with Decisions, a current `--operator-clear-front` site report, a finite test duration, and the exact `--driver-pid`, `--driver-exe`, and `--driver-sha256`. It uses a separate relay bound to the candidate driver's watchdog. The relay accepts ordinary zero setpoints and keeps enforcing controller heartbeat and driver status. `--max-travel-m` bounds the supervised run using cumulative wheel odometry. The hardware profile offers forward and stop and emits zero angular velocity. Its JSON contains the relay-attributed `/cmd_vel` trace, the longest uninterrupted nonzero interval, and wheel-odometry displacement.

`--drive-arcs` selects the hardware profile for forward segments and shallow turns. It requires Decisions, the same driver identity and a finite duration. Each direction must pass the current grid's swept-body check. The relay bounds forward speed and steering curvature and stops on lost heartbeat. The run records angular commands and the measured heading change of each model-selected arc by decision ID. Turn validation needs observed side and rear clearance. Use the private preview to check which directions the installed sensors support.

`--static-room` enables an opt-in route for a supervised room with fixed surroundings. Free observations up to 30 seconds old can fill an occluded cell next to or behind the chassis. The leading corridor needs live observations. Isolated occupied readings can remain for 1.5 seconds. Confirmed depth hits keep their height interval after laser free, occlusion or age and are cleared only by fresh matching depth volume evidence. The current native snapshot produces positive depth evidence. Its heights are in the odom frame, and producing a clearing volume requires a calibrated depth path. Build and source the same `FreshGrid` message on both the publisher and subscriber.

In this mode Decisions selects a leftward or rightward route. The local controller checks each short segment before continuing on that route. When a turn is occupied, the controller can take a separately checked turn away from it or a checked straight segment. At a blocked start, Decisions can choose a pivot if both the rotation and its exit pass the swept-body checks. The controller reviews an active route asynchronously on a newly checked segment after 45 seconds and keeps tracking while the request is pending. Timeline entries with `stage=local_continuation` link a locally checked segment to its source model decision. The result reports wheel-odometry heading change, return distance and continuous command coverage for a measured lap.

The room output thread sends zero when the latest command has passed its sensor acquisition deadline. It keeps the relay heartbeat alive during a delayed geometry update. Losing the entire navigation process still trips the independent relay watchdog. The relay-attributed `/cmd_vel` trace records what the base received.

For an attended room, `--roam` with `--drive-arcs --static-room` keeps the local planner running until shutdown. An optional `--seconds` or `--max-travel-m` bounds a commissioning run. A lost relay heartbeat first stops the old relay. The next relay starts only after the old process has exited, distinct recent wheel samples confirm a stop, and fresh observations and command-ownership checks pass. The planner keeps its room observations and checks every new motion against the current grid. Competing command sources and driver faults end the session. With no verified route, the planner continues examining new observations at zero speed.

Pair `--roam` with the recorder's `--rolling` mode to rotate compact ROS bags within the specified budget. An interrupt to the foreground recorder shuts down motor output and finalizes the recording.

The roaming controller wakes on fresh grids and checks delayed odometry for continuity before it can use the next fresh pose. After settling from a blocked turn, it checks new routes from the measured pose. If full starts and observation turns are unavailable, it can offer shorter measured segments with the same steering and stopping-clearance checks. The decision targets and timeline include these distances.

The room profile checks 12 cm segments and looks farther for occupied cells when selecting a detour. A point turn is available when the straight route encounters an obstacle, the complete rotational sweep is observed, and a checked forward exit is available. The room relay caps rotation to 0.05 rad/s and enforces the same command and driver watchdogs during a point turn.

The room controller follows through to the next checked segment as it nears a waypoint. While moving, it periodically checks for measured obstacles up to 36 cm along the current route, including the prepared successor. When its immediate sweep is still clear, it can start a checked detour at the measured pose and retain its forward speed. The nearer detour sweep must have live coverage, and the longer sweep must be clear of measured obstacles. The tracker checks the actual steering before publishing. The timeline records these changes as `early_detour` events.

If the current drive arc becomes occupied, the controller can publish a different arc from the measured pose in the same control cycle. This requires fresh coverage of the replacement segment and no measured obstacle in its longer sweep. The controller keeps its forward speed through that change and records `occupied_detour`. It commands zero when these checks cannot establish a replacement.

After an early turning detour, a checked successor can continue the turn for up to 24 cm or 20 seconds while the model's route stays the same. Each successor needs its own swept-body check. The timeline records these successors as `detour_continuation`. On lost coverage the controller checks an alternative and its steering transition before publishing its single command for that cycle. It records `unknown_detour` for an accepted replacement and sends zero if clearance remains unverified. Resuming the original route requires 0.4 seconds of continually clear geometry across fresh lidar and depth observations.

Curvature changes are gradual when the resulting sweep is clear. A point turn can prepare and hand off to its checked forward exit. If path tracking loses the waypoint, the controller stops the current segment and can prepare a new segment at the measured pose using the accepted route. Clock and odometry faults still cancel the route.

At a stationary corner with an observed obstruction and no checked start, the controller can test wider point turns after recent wheel odometry confirms that the robot has settled. The full rotation and forward exit must pass the same sweep checks. Each rotation command is checked again against the current view. The timeline records the selected angle as `turn_target_rad`.

`run_trail_recovery.py` checks a recorded trail when the robot has stopped close to an obstacle. Its default verifies the recorded wheel odometry and reference without connecting to hardware. With `--recorded-curve` it can check a trail that includes measured turns. The supervised execute mode matches the live forward reference and tracks the recorded path with wheel odometry. The reverse relay enforces separate limits on speed, turn rate, and motion duration. This recovery depends on the trail remaining clear in the attended static room.

With `--recorded-curve`, `--start-at-seq` selects an odometry sample inside a completed run when recovering from the middle of a track. `--prior-trace` can extend the track through an identical stopped pose from an earlier run on the same driver. The recorder keeps the original timestamps and reports the gap between recordings. The live odometry and current forward scan must match the starting reference before the relay can move.

Recorded recovery accepts measured strips from 4 to 45 cm, including a shorter final span between odometry keyframes. It checks front laser returns against the padded footprint swept by both the commanded translation and rotation over the stopping horizon. A front obstruction commands zero immediately. If fresh scans clear the same trail for 0.35 seconds, the follower can resume from the measured pose. Without renewed clearance within two seconds, recovery ends.

## Exploration memory preview

Launch the ROS Melodic RTAB-Map profile once, using a database paired with a new map ID and a new memory file for the first session. Store all three in the same session directory. An existing map needs the original pair and confirmed relocalization before its saved locations are projected into odom.

```sh
mkdir -p "$HOME/jev-room-map"
roslaunch standalone/exploration_mapping.launch database_path:="$HOME/jev-room-map/room.db"
```

With the native sensing overlay sourced, preview the combined pipeline in another shell.

```sh
/usr/bin/python2 standalone/run_navigation.py --static-room --decisions --seconds 60 \
  --map-id room-a --spatial-memory "$HOME/jev-room-map/room.json" \
  --timeline "$HOME/jev-room-map/preview.jsonl"
```

The mapper uses wheel odometry and the fixed forward laser. RGB-D remains on the existing local obstacle path. The controller stays in continuous odom and reprojects keyframe-anchored obstacles after map corrections. The Decisions worker receives IDs for currently checked endpoints, visit counts and a short-range frontier estimate. The preview publishes on its private command topic. After moving the robot by hand, treat the new start as a new session. If the old map cannot be matched, choose a new database, map ID and memory file.

An optional qualified vision or scan-matching `nav_msgs/Odometry` feed can be passed with `--independent-odom-topic` during supervised driving. The progress monitor integrates relay-attributed output, checks wheel and independent movement, and stops on insufficient progress or inconsistent localization. Recorded recovery refuses faulted trails and runs with recorded low-obstacle or map-backed hazards until it can check those hazards itself.

To include long-term notes, add `--powercontext-url` and `--powercontext-scope` for the room's PowerContext service. The optional `POWERCONTEXT_TOKEN` is read from the process environment. The navigation worker requests scoped history asynchronously. The event writer records selected outcomes and proposes Experience entries for measured, map-anchored no-progress reports. Candidate review uses the source and candidate IDs in the logs. PowerContext availability does not gate local motion checks.

## Recorded sessions

Use the recording entry point with any standalone preview or supervised navigation arguments. The output filesystem needs room for the configured bag budget plus 256 MiB of reserve.

```sh
mkdir -p /path/to/session-recordings
/usr/bin/python3 standalone/record_navigation.py --output-root /path/to/session-recordings -- \
  --static-room --decisions --seconds 30
```

Each invocation prints a new recording directory. It contains `timeline.jsonl`, the `navigation.json` result with wheel motion and relay attribution for driving runs, and segmented LZ4 ROS bags. The bags record actual `/cmd_vel`, odometry, scan, depth calibration, transforms, the control grid and the RTAB-Map pose graph and status when available. Depth is sampled once every two seconds. Add `--extra-topic` before `--` to include a separate measured pose feed, IMU or other auxiliary evidence. The manifest stores a shared namespace, revision, wall and monotonic clock anchors, and both process exit statuses. The recorder retains completed segments and stops navigation if recording fails or exhausts its budget.

Add `--map-database /path/to/room.db` before `--` on platforms whose wheel driver leaves global TF unpublished. This starts RTAB-Map on the navigation session's private odometry TF stream. Pass the matching `--map-id` and `--spatial-memory` after `--`.

For an attended restart in the same static room, `--room-revisit` accepts a recorded blind-side observation mask with its stop pose, laser reference and driver generation. The mask contributes only after registration against live odometry and laser observations. The controller can use paths already checked by the live view while registration is pending. Current occupancy invalidates the relevant cell. The mask retires after the chassis travels 30 cm from the registered pose, and each cell retains the age limit of its source observation. Near a measured obstacle, a stationary observation turn may be offered if its complete rotation is clear. The controller checks translation after the turn.

## Bounded forward trial

This is a diagnostic implementation that communicates with ROS 1 directly;
it does not require the `rosclawd` service. `supervised_trial.py` defaults to
observation only. Its explicit `--execute` mode obtains one Decisions API
choice from a camera image and sensor state, and only streams forward commands
if that choice is `forward` and the local checks pass. It does not implement
turning, reverse motion, a persistent exploration loop, or autonomous mapping.

The ordinary trial has a 0.3-second relay motion deadline. The explicit
`--longer-forward` option selects a separate, bounded two-second forward trial.
Adding `--single-threaded` to that option selects an experimental synchronous
sampling loop with the same motion bounds and per-command freshness checks.
Both require fresh camera, depth, lidar, odometry and base status, current ROS
and serial-ownership observations, an operator site report, and an explicit
floor-height and camera-rectification assumption. A dedicated relay publishes
velocity heartbeats and stops when its intent expires, its parent disappears,
the driver watchdog fails, or the selected motion deadline elapses. The relay
also receives the earliest absolute sensor, model and monitor expiry with each
motion intent. The relay and the candidate driver do not prove that the
physical chassis stopped.

The depth projection uses the supplied floor-height assumption and a rigid
camera transform; neither is independently calibrated by this code. The lidar
sees only part of the surroundings, and gaps, occlusions, the camera near
field and areas outside the sensors' view remain unknown. The camera TF
fallback only accepts a nearby latest transform matching a previously
observed timestamped transform. The first model decision is not renewed
during a single trial. A fresh full image is supplied for the model; later
safety snapshots continue checking projected depth and lidar without repeated
JPEG encoding or unused image-grid statistics. A newer depth frame must be
projected and checked; a newer lidar frame must have its own measured sampled
cross-section coverage before an older fused sample can still be used. The
supervised runner defers the adapter's intermediate current-frame check to
avoid duplicate work; it still rechecks current sensors immediately before
each motion command. Intent validity also ends when historical rays required
for the accepted coverage expire. If snapshots cannot be produced and
checked fast enough, the relay stops instead of continuing an expired intent;
smooth real-robot operation has not yet been established.

The longer supervised trial optionally refreshes observation history while
the model request is pending; that worker cannot command motion and must exit
before the trial resumes observation. A worker that fails to exit prevents
motion. Expensive current-sensor checks use a captured view rather than holding
the ROS callback lock. They recheck the latest odometry, state, transforms and
sensor generations before returning, and cap any newer-scan verdict by the
expiry of the historical rays on which it depended. Unchecked later scans or
depth frames still reject the result.

The optional Cython scan accelerator can be built for ROS Melodic Python 2
with `/usr/bin/python2 standalone/build_scan_accel.py build_ext --build-lib
standalone/build/py2 --build-temp standalone/build/temp-py2`. Start a new
controller process after building. The binary is local and excluded from git;
without it, the existing checked Python implementation is used. Geometric
threshold ties and inputs outside the accelerated format also use that
implementation. This optimization does not establish physical clearance.
The accelerator also projects new raw scans and computes the expiry of an
already checked forward cross-section. Unusual input types and boundary ties
fall back to the scalar checks. The expiry accelerator bounds its work to
2,048 rays and eight distinct acquisition timestamps; larger histories use
the scalar implementation. These changes do not extend sensor freshness.
Fusion coordinate transforms use batches of up to 1,536 rays per native call.
A single record beyond that bound or a declined batch triggers the checked
scalar path for the whole snapshot, including its historical obstacles.
For the longer supervised trial, observation production and final validation
are serialized with a bounded wait. A failed observation or assessment blocks
new proofs; hard failures latch for the trial. The final proof and relay pipe
write share the same processing guard. Sensor callbacks remain independent,
and the relay still enforces each absolute intent expiry.
The guarded longer trial starts its sampler while the relay is still at zero,
waits for an assessed sample and a sufficiently long first intent, and retries
temporarily incomplete forward coverage only while stationary. Once motion
starts, the same incomplete coverage latches the trial to stop.

The November 20 perception and renewal work adds **read-only diagnostics**:
`sensing_timeline.py` and `live_independent_probe.py` record acquisition,
queueing and processing times, while `multiprocess_probe.py` runs independent
scan and depth workers and an in-memory `RelayGate`. The latter does not
publish velocity or check base status, odometry continuity or clearance.
Its `timeline_complete`/`mux_writer_complete` field refers to the mux writer
only; child shutdown events can be present only in the child timelines.
The immutable cache and local cell evidence modules are experimental and are
not connected to a motion controller. Sampled centers do not prove continuous
3-D coverage or a swept-footprint clearance. Full scan/depth coverage and
independent fault propagation remain open problems.

The supervised runner's opt-in `--longer-forward --prepared-forward` mode
performs projection and full sensor validation in the existing background
worker. Before submitting, the controller checks the original proof expiry,
current sensor generations, odometry, base state and command monitors. It
withholds commands if a new frame supersedes the prepared proof. The first live
tests of this mode completed at zero because proofs were repeatedly superseded.
See the [November 20 trial records](../docs/2026-11-20-perception-renewal-progress.md).

Add `--preproject-depth` to the prepared mode to project depth on a dedicated
worker. This experiment checks the original frame identity, its calibration
and its transform before reusing a projection. The perception worker waits
up to 20 ms for an exact cache result during the final depth check. A missing
projection cannot authorize a command.

Use `--split-sensor-checks` with prepared mode to run scan and depth checks
on separate workers. The first command uses the synchronous check and
seeds the workers with its existing deadline. Each subsequent command
checks current state and uses the earliest applicable expiry. At most
two pending arrivals per sensor can be tolerated within the previous
proof, with its deadline capped by the oldest pending receipt and
the 70 ms renewal budget. A pending frame contributes no clearance.
Confirmed obstacles latch the split worker fault. This remains an opt-in
experiment and has yet to sustain motion on the robot.

Add `--depth-projection-process` to the split checks with preprojection
enabled to isolate projection from the Python process that receives ROS
messages and renews commands. The child receives the exact depth buffer via
a private socket and uses the existing projection. It does not subscribe to
ROS or publish velocity. The single depth producer fills the same cache.
An out-of-budget projection supplies no evidence. A stalled transport closes
without refreshing any acquisition timestamp, and the local relay still
enforces the original expiry.

`continuous_policy.py` is an offline-only forward-intent prototype. It is not
wired to the real relay. `sensor_ipc.py` and `depth_ray_coverage.py` are also
offline prototypes, not a running sensor worker or permission to merge blind
regions. `relay_registry.py` provides optional, narrowly
validated cleanup for a dead relay's stale ROS registrations; its default mode
does not mutate the ROS graph. The candidate source changes for the base
watchdog and the YDLIDAR diagnostic driver are in `patches/`; applying a
patch does not install or start either driver. In the current development
session the candidate base binary was launched from `/tmp`, so its continued
use after reboot or process restart has not been established.

For offline verification, run the standalone tests with Python 2.7 in a ROS
Melodic environment, and run the repository's non-live suite with Python 3.
`threaded_relay_pty_check.py` exercises a real relay and candidate driver
against a fake serial chassis only inside its documented isolated namespace;
its results are not physical stopping or clearance tests.

`native_sensing_probe.py` is a separate, bounded observation experiment. It
creates a private TF topic and rolling grid using ROS native lidar and depth
processing, without publishing `/tf` or `/cmd_vel`. It does not need a saved
map. The optional `--test-synthetic-obstacle` injects a point only into the
private probe topic to check obstacle marking, then discards the grid. The
private TF topic derives `odom → base_link` from odometry, since the current
base configuration does not publish that transform. The standard obstacle
layer can retain old free cells. The optional experimental native layer resets
laser free space on each update and exposes the final grid and the scan's
acquisition timestamp in a single `FreshGrid` message. It requires a recent
scan and rejects unsafe laser and unknown-space configurations. A delayed scan
older than the last accepted scan is rejected, including while it is within
the age limit. After a clock or sensor timestamp reset, scans remain unavailable
until their timestamps catch up or the layer is restarted. The probe uses a
conventional marking-only depth layer. The navigation preview selects
`FreshDepthLayer` to bind the depth acquisition time to the same grid as the
scan. The depth height thresholds and camera rectification remain calibration
assumptions. The preview preserves unknown cells in its footprint checks.

To build the experimental layer, create a catkin workspace with
`standalone/limo_fresh_layer` linked or copied into its `src` directory and
run `catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2`. Pass that workspace to
`native_sensing_probe.py --fresh-layer-workspace PATH` to test live sensor
inputs without motion. `fresh_laser_integration_probe.py --workspace PATH`
checks behavior using private synthetic ROS topics; `--unsafe-inf-config` and
`--unsafe-unknown-config` test that invalid configurations are rejected. The
build is local to the chosen catkin workspace and is not installed by these
commands.
