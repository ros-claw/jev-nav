#include "limo_command_watchdog.h"

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>

int main() {
  LimoCommandWatchdog gate(0.3);
  assert(gate.should_stop(10.0));
  assert(!gate.is_latched());
  assert(!gate.accept(0.1, 0.0, 0.0, 10.0));
  assert(gate.accept(0.0, 0.0, 0.0, 10.0));
  gate.record_write(true, 10.0);
  assert(gate.accept(0.1, 0.0, 0.0, 10.01));
  gate.record_write(true, 10.01);
  assert(!gate.should_stop(10.30));
  assert(gate.should_stop(10.32));
  assert(gate.is_latched());
  assert(!gate.accept(0.1, 0.0, 0.0, 10.32));
  assert(gate.should_stop(10.33));

  LimoCommandWatchdog clean_stop(0.3);
  assert(clean_stop.accept(0.0, 0.0, 0.0, 20.0));
  clean_stop.record_write(true, 20.0);
  assert(clean_stop.accept(0.1, 0.0, 0.0, 20.01));
  clean_stop.record_write(true, 20.01);
  assert(clean_stop.accept(0.0, 0.0, 0.0, 20.1));
  clean_stop.record_write(true, 20.1);
  assert(!clean_stop.should_stop(21.0));
  assert(!clean_stop.is_latched());
  assert(clean_stop.accept(0.1, 0.0, 0.0, 21.1));

  LimoCommandWatchdog failed_zero(0.3);
  assert(failed_zero.accept(0.0, 0.0, 0.0, 25.0));
  failed_zero.record_write(true, 25.0);
  assert(failed_zero.accept(0.1, 0.0, 0.0, 25.01));
  failed_zero.record_write(true, 25.01);
  assert(failed_zero.accept(0.0, 0.0, 0.0, 25.02));
  failed_zero.record_write(false, 25.02);
  assert(failed_zero.is_latched());
  assert(!failed_zero.accept(0.1, 0.0, 0.0, 25.03));

  LimoCommandWatchdog late(0.3);
  assert(late.accept(0.0, 0.0, 0.0, 30.0));
  late.record_write(true, 30.0);
  assert(late.accept(0.1, 0.0, 0.0, 30.01));
  late.record_write(true, 30.01);
  assert(!late.accept(0.1, 0.0, 0.0, 30.32));
  assert(late.should_stop(30.32));

  LimoCommandWatchdog invalid(0.3);
  assert(!invalid.accept(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 1.0));
  assert(invalid.should_stop(1.0));

  LimoCommandWatchdog backwards(0.3);
  assert(backwards.accept(0.0, 0.0, 0.0, 5.0));
  backwards.record_write(true, 5.0);
  assert(backwards.accept(0.1, 0.0, 0.0, 5.01));
  backwards.record_write(true, 5.01);
  assert(backwards.should_stop(4.0));

  LimoCommandWatchdog blocked_write(0.3);
  assert(blocked_write.accept(0.0, 0.0, 0.0, 40.0));
  blocked_write.record_write(true, 40.0);
  assert(blocked_write.accept(0.1, 0.0, 0.0, 40.01));
  blocked_write.record_write(true, 40.01);
  assert(blocked_write.accept(0.0, 0.0, 0.0, 40.20));
  blocked_write.record_write(true, 40.40);
  assert(blocked_write.is_latched());
  assert(!blocked_write.accept(0.1, 0.0, 0.0, 40.41));

  // A zero can prime the driver before the first chassis mode report, but a
  // motion command must wait for a supported mode without latching at startup.
  LimoCommandWatchdog mode(0.3);
  assert(mode.accept(0.0, 0.0, 0.0, 50.0, false));
  mode.record_write(true, 50.0);
  assert(mode.primed());
  assert(mode.zero_writes() == 1);
  assert(!mode.accept(0.1, 0.0, 0.0, 50.01, false));
  assert(!mode.is_latched());
  assert(mode.accept(0.1, 0.0, 0.0, 50.02, true));
  mode.record_write(true, 50.02);
  assert(!mode.accept(0.1, 0.0, 0.0, 50.03, false));
  assert(mode.is_latched());
  assert(!mode.accept(0.1, 0.0, 0.0, 50.04, true));

  LimoCommandWatchdog mode_change(0.3);
  assert(!mode_change.mode_changed());
  assert(mode_change.accept(0.0, 0.0, 0.0, 60.0));
  mode_change.record_write(true, 60.0);
  assert(mode_change.accept(0.1, 0.0, 0.0, 60.01));
  mode_change.record_write(true, 60.01);
  assert(mode_change.mode_changed());
  assert(mode_change.is_latched());
  assert(!mode_change.accept(0.1, 0.0, 0.0, 60.02));

  LimoCommandWatchdog publisher(0.3);
  assert(publisher.accept_source("/relay", true));
  assert(publisher.accept(0.0, 0.0, 0.0, 70.0));
  publisher.record_write(true, 70.0);
  publisher.record_zero_source("/relay");
  assert(publisher.accept_source("/relay", false));
  assert(publisher.accept(0.1, 0.0, 0.0, 70.01));
  publisher.record_write(true, 70.01);
  assert(!publisher.accept_source("/other", true));
  assert(publisher.is_latched());

  LimoCommandWatchdog next_task(0.3);
  assert(next_task.accept_source("/relay_a", true));
  assert(next_task.accept(0.0, 0.0, 0.0, 80.0));
  next_task.record_write(true, 80.0);
  next_task.record_zero_source("/relay_a");
  assert(next_task.accept_source("/relay_b", true));
  assert(next_task.accept(0.0, 0.0, 0.0, 80.01));
  next_task.record_write(true, 80.01);
  next_task.record_zero_source("/relay_b");
  assert(!next_task.accept_source("/relay_a", false));
  assert(next_task.is_latched());
  std::cout << std::fixed << std::setprecision(9)
            << std::chrono::duration<double>(
                   std::chrono::steady_clock::now().time_since_epoch()).count() << '\n';
  return 0;
}
