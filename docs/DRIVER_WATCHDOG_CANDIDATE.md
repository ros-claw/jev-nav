# Candidate LIMO base command watchdog

The patch `patches/limo-base-command-watchdog.patch` is an **uninstalled
candidate** for the on-device `limo_base` ROS 1 driver. It was generated against
the observed local source tree at AgileX `limo_ros` commit `4d7dc93` together
with that tree's existing local driver and launch changes. It is not a generic
patch for every LIMO driver version. Check it against the exact target source;
do not overwrite existing local modifications.

With the optional `command_watchdog_enabled` launch argument set to `true`, the
driver requests zero before it has accepted a zero input and requires a complete
serial write of an incoming zero command before accepting a nonzero command.
After a nonzero command, a monotonic 0.3-second default timeout latches the gate,
requests zero repeatedly, and rejects later nonzero commands. A short or failed
serial write, or a write that finishes after the preceding motion deadline,
also latches the gate. A successfully written zero command ends
the active command period, so a later task may start without restarting the
driver. Nonzero motion also waits for a supported, recent chassis mode report;
the serial reader and command handler serialize mode changes with motion writes.
If the reported mode changes or becomes stale while a motion command is active,
the watchdog latches and requests zero. The proposed 0.5-second mode freshness
limit is not measured for the target hardware. Timestamps are captured before
the driver mutex; they do not detect frames that already sat in the device or
kernel receive buffer.

When enabled, the candidate records the caller name on a completed zero write.
A different caller during motion, or a nonzero command from a different caller,
latches the watchdog; while idle, a new caller may send zero to begin a new
task. This guards against accidental mixing but does not authenticate a caller.
The timeout and 50 ms polling period are initial test values, not
measured stopping guarantees. The feature defaults to disabled to avoid
changing existing deployments before integration and testing.

`worker/limo_command_watchdog.h` is the standalone policy used by the patch;
`tests/test_command_watchdog.py` compiles and exercises it without ROS or a
robot. The patched source has also been compiled in an isolated ROS Melodic
workspace. No patched driver has been installed or physically exercised.

This watchdog lives in the same process and ROS callback queue as the serial
driver. It cannot stop the robot if that process, callback queue, serial write,
computer or power fails. ROS 1 `Twist` lacks a timestamp and publisher identity:
the watchdog cannot distinguish delayed commands or another publisher that
continues sending. Restarting the driver resets its in-memory latch. A complete
real deployment still needs command-source arbitration, the daemon-owned
executor and authorization path, an independent chassis command-loss stop, and
physical verification. A complete serial write is not a controller
acknowledgement or proof of physical stopping.

The candidate relay sends zero until a stable ROS subscriber connection and a
fresh status from the candidate driver's `/rosclaw_limo/command_watchdog` topic
show an increased completed-zero-write counter with the relay's ROS caller name.
The status carries a source monotonic timestamp and must arrive fresh on the
same host. The relay stops on stale or
latched status, a reported mode change, a driver restart, or loss of its ROS
subscriber. The caller name is ROS 1 connection metadata, not an authenticated
identity: other nodes can impersonate it or the status, or inject commands.
Source arbitration is still required before live use. A completed serial write
does not prove chassis receipt.
`MOTION_ACK` means that a nonzero command was
published while a subscriber was connected; it does not prove that the driver
accepted it or that the chassis moved. Mode changes may also occur physically
before their status frame reaches the driver; `use_mcnamu` overrides reported
mode and requires separate verification.
