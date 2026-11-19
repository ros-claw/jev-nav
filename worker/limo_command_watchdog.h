#ifndef ROSCLAW_LIMO_COMMAND_WATCHDOG_H
#define ROSCLAW_LIMO_COMMAND_WATCHDOG_H

// Candidate driver-side timeout policy. No ROS or hardware access. Calls must
// be serialized with the driver's motion-command writes by the caller.

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

class LimoCommandWatchdog {
 public:
  explicit LimoCommandWatchdog(double timeout_seconds, double mode_timeout_seconds = 1.5)
      : timeout_(timeout_seconds), mode_timeout_(mode_timeout_seconds), last_(0.0), seen_(false), active_(false),
        primed_(false), pending_(false), pending_zero_(false), prior_deadline_(0.0),
        prior_active_(false), latched_(false), zero_writes_(0), last_zero_write_at_(0.0) {
    if (!std::isfinite(timeout_) || timeout_ < 0.1 || timeout_ > 1.0) {
      throw std::invalid_argument("command timeout must be between 0.1 and 1.0 seconds");
    }
    if (!std::isfinite(mode_timeout_) || mode_timeout_ < 0.7 || mode_timeout_ > 5.0) {
      throw std::invalid_argument("mode timeout must be between 0.7 and 5.0 seconds");
    }
  }

  bool accept(double linear, double lateral, double angular, double now,
              bool supported_mode = true) {
    if (latched_) return false;
    if (pending_ || !std::isfinite(linear) || !std::isfinite(lateral) || !std::isfinite(angular) ||
        !std::isfinite(now) || (seen_ && (now < last_ || (active_ && now - last_ >= timeout_)))) {
      latched_ = true;
      return false;
    }
    const bool zero = linear == 0.0 && lateral == 0.0 && angular == 0.0;
    if (!primed_ && !zero) return false;
    if (!supported_mode && !zero) {
      if (active_) latched_ = true;
      return false;
    }
    prior_active_ = active_;
    prior_deadline_ = last_ + timeout_;
    last_ = now;
    seen_ = true;
    pending_ = true;
    pending_zero_ = zero;
    if (!zero) active_ = true;
    return true;
  }

  void record_write(bool complete, double now) {
    if (!complete || !pending_ || !std::isfinite(now) || now < last_ ||
        now - last_ >= timeout_ || (prior_active_ && now >= prior_deadline_)) {
      latched_ = true;
    } else if (pending_zero_) {
      if (zero_writes_ == std::numeric_limits<std::uint64_t>::max()) {
        latched_ = true;
      } else {
        ++zero_writes_;
        last_zero_write_at_ = now;
        active_ = false;
        primed_ = true;
      }
    }
    pending_ = false;
    pending_zero_ = false;
  }

  bool should_stop(double now) {
    if (!std::isfinite(now) || (seen_ && (now < last_ || (active_ && now - last_ >= timeout_)))) {
      latched_ = true;
    }
    return latched_ || !primed_;
  }

  bool is_latched() const { return latched_; }
  bool primed() const { return primed_; }
  std::uint64_t zero_writes() const { return zero_writes_; }
  double last_zero_write_at() const { return last_zero_write_at_; }
  const std::string& zero_source() const { return zero_source_; }

  bool mode_report_fresh(double report_at, double now) const {
    return std::isfinite(report_at) && std::isfinite(now) && report_at > 0.0 &&
           now >= report_at && now - report_at < mode_timeout_;
  }

  bool accept_source(const std::string& source, bool zero) {
    if (latched_) return false;
    if (source.empty() || source[0] != '/' || source.size() > 80 ||
        source.find_first_of(" \t\r\n") != std::string::npos ||
        (primed_ && source != zero_source_ && (active_ || !zero))) {
      latched_ = true;
      return false;
    }
    return true;
  }

  void record_zero_source(const std::string& source) {
    if (primed_ && !latched_) zero_source_ = source;
  }

  bool mode_changed() {
    if (active_) latched_ = true;
    return latched_;
  }

 private:
  const double timeout_;
  const double mode_timeout_;
  double last_;
  bool seen_;
  bool active_;
  bool primed_;
  bool pending_;
  bool pending_zero_;
  double prior_deadline_;
  bool prior_active_;
  bool latched_;
  std::uint64_t zero_writes_;
  double last_zero_write_at_;
  std::string zero_source_;
};

#endif
