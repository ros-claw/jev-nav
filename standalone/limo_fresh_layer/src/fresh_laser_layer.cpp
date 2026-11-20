// Experimental sensing layer only: not a swept-volume or stopping guarantee.
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <costmap_2d/cost_values.h>
#include <costmap_2d/obstacle_layer.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/TimeReference.h>
#include <limo_fresh_layer/FreshGrid.h>

namespace limo_fresh_layer {

class BufferLock {
 public:
  explicit BufferLock(costmap_2d::ObservationBuffer* buffer) : buffer_(buffer) { buffer_->lock(); }
  ~BufferLock() { buffer_->unlock(); }
  BufferLock(const BufferLock&) = delete;
  BufferLock& operator=(const BufferLock&) = delete;

 private:
  costmap_2d::ObservationBuffer* buffer_;
};

class FreshLaserLayer : public costmap_2d::ObstacleLayer {
 public:
  FreshLaserLayer() {
    // The base destructor checks dsrv_ even if initialization is refused.
    dsrv_ = nullptr;
    enabled_ = false;
    current_ = false;
  }

  void onInitialize() override {
    ros::NodeHandle private_node("~/" + name_);
    private_node.param("max_scan_age", max_scan_age_, 0.3);
    std::string sources, source, extra, data_type;
    private_node.param("observation_sources", sources, std::string(""));
    std::istringstream source_stream(sources);
    const bool exactly_one_source = static_cast<bool>(source_stream >> source) && !(source_stream >> extra);
    bool inf_is_valid = true, marking = false, clearing = false, footprint_clearing = true;
    double persistence = -1.0;
    ros::NodeHandle source_node(private_node, source);
    source_node.param("data_type", data_type, std::string(""));
    source_node.param("inf_is_valid", inf_is_valid, false);
    source_node.param("marking", marking, true);
    source_node.param("clearing", clearing, false);
    source_node.param("observation_persistence", persistence, 0.0);
    private_node.param("footprint_clearing_enabled", footprint_clearing, true);
    valid_config_ = std::isfinite(max_scan_age_) && max_scan_age_ > 0 && max_scan_age_ <= 0.3
                    && layered_costmap_->isTrackingUnknown() && exactly_one_source
                    && data_type == "LaserScan" && !inf_is_valid && marking && clearing
                    && persistence == 0.0 && !footprint_clearing;
    if (!valid_config_) {
      ROS_FATAL("fresh laser layer configuration is unsafe or unsupported");
      ros::shutdown();
      return;
    }
    costmap_2d::ObstacleLayer::onInitialize();
    setDefaultValue(costmap_2d::NO_INFORMATION);
    evidence_publisher_ = private_node.advertise<sensor_msgs::TimeReference>("scan_evidence", 1);
  }

  void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y,
                    double* max_x, double* max_y) override {
    // Only one buffer, used for both marking and clearing, is supported.
    // Holding its recursive lock prevents a new unchecked scan from racing
    // the timestamp check and the inherited processing of that observation.
    if (valid_config_ && enabled_ && clearing_buffers_.size() == 1 && marking_buffers_.size() == 1
        && clearing_buffers_[0] == marking_buffers_[0]
        && static_clearing_observations_.empty() && static_marking_observations_.empty()) {
      BufferLock guard(clearing_buffers_[0].get());
      std::vector<costmap_2d::Observation> observations;
      bool fresh = getClearingObservations(observations) && observations.size() == 1;
      if (fresh) {
        const ros::Time stamp = observations[0].cloud_->header.stamp;
        const double age = (ros::Time::now() - stamp).toSec();
        // A delayed scan must not replace a newer accepted observation.
        fresh = !stamp.isZero() && stamp >= accepted_scan_stamp_
                && std::isfinite(age) && age >= 0 && age <= max_scan_age_;
      }
      resetMaps();
      if (fresh) {
        accepted_scan_stamp_ = observations[0].cloud_->header.stamp;
        costmap_2d::ObstacleLayer::updateBounds(robot_x, robot_y, robot_yaw, min_x, min_y, max_x, max_y);
        const double final_age = (ros::Time::now() - observations[0].cloud_->header.stamp).toSec();
        if (!std::isfinite(final_age) || final_age < 0 || final_age > max_scan_age_) {
          resetMaps();
          current_ = false;
        }
      } else {
        if (rolling_window_) updateOrigin(robot_x - getSizeInMetersX()/2, robot_y - getSizeInMetersY()/2);
        current_ = false;
      }
    } else {
      if (rolling_window_) updateOrigin(robot_x - getSizeInMetersX()/2, robot_y - getSizeInMetersY()/2);
      resetMaps();
      current_ = false;
    }
    // Reset the entire master region too; otherwise a previous free cell can
    // outlive the laser observation after the local layer is reset.
    *min_x = std::min(*min_x, getOriginX());
    *min_y = std::min(*min_y, getOriginY());
    *max_x = std::max(*max_x, getOriginX() + getSizeInMetersX());
    *max_y = std::max(*max_y, getOriginY() + getSizeInMetersY());
  }

  void updateCosts(costmap_2d::Costmap2D& master, int min_i, int min_j, int max_i, int max_j) override {
    const ros::Time checked_at = ros::Time::now();
    const double age = (checked_at - accepted_scan_stamp_).toSec();
    const bool fresh = valid_config_ && enabled_ && current_ && !accepted_scan_stamp_.isZero()
                       && std::isfinite(age) && age >= 0 && age <= max_scan_age_;
    if (!fresh) {
      resetMaps();
      current_ = false;
      // CostmapLayer's helper returns early when disabled; a disabled or
      // stale layer must not leave this region marked free by the master.
      unsigned char* data = master.getCharMap();
      const unsigned int width = master.getSizeInCellsX();
      for (int j = min_j; j < max_j; ++j) {
        for (int i = min_i; i < max_i; ++i) {
          data[static_cast<unsigned int>(j) * width + static_cast<unsigned int>(i)] = costmap_2d::NO_INFORMATION;
        }
      }
    } else {
      costmap_2d::ObstacleLayer::updateCosts(master, min_i, min_j, max_i, max_j);
    }
    // Header time is the check before the grid is published, and time_ref is
    // the actual scan acquisition time. This is diagnostic evidence, not a
    // motion permit; consumers must bind it to the corresponding grid.
    sensor_msgs::TimeReference evidence;
    evidence.header.stamp = checked_at;
    evidence.time_ref = fresh ? accepted_scan_stamp_ : ros::Time(0);
    evidence.source = fresh ? "fresh_scan" : "unavailable";
    if (evidence_publisher_) evidence_publisher_.publish(evidence);
  }

  bool scanFreshAt(const ros::Time& at, ros::Time* stamp) const {
    const double age = (at - accepted_scan_stamp_).toSec();
    const bool fresh = valid_config_ && enabled_ && current_ && !accepted_scan_stamp_.isZero()
                       && std::isfinite(age) && age >= 0 && age <= max_scan_age_;
    if (stamp) *stamp = fresh ? accepted_scan_stamp_ : ros::Time(0);
    return fresh;
  }

 private:
  double max_scan_age_ = 0.3;
  bool valid_config_ = false;
  ros::Time accepted_scan_stamp_;
  ros::Publisher evidence_publisher_;
};

// Must be the last layer: publishes the complete grid and the scan stamp
// together while LayeredCostmap holds the master-grid lock.
class FreshGridSnapshotLayer : public costmap_2d::Layer {
 public:
  FreshGridSnapshotLayer() { current_ = true; enabled_ = true; }

  void onInitialize() override {
    ros::NodeHandle private_node("~/" + name_);
    publisher_ = private_node.advertise<limo_fresh_layer::FreshGrid>("fresh_grid", 1);
  }

  void updateBounds(double, double, double, double*, double*, double*, double*) override {}

  void updateCosts(costmap_2d::Costmap2D& master, int, int, int, int) override {
    if (!enabled_ || !publisher_ || publisher_.getNumSubscribers() == 0) return;
    const ros::Time now = ros::Time::now();
    const std::vector<boost::shared_ptr<costmap_2d::Layer> >* plugins = layered_costmap_->getPlugins();
    FreshLaserLayer* laser = nullptr;
    size_t count = 0;
    for (const auto& plugin : *plugins) {
      FreshLaserLayer* candidate = dynamic_cast<FreshLaserLayer*>(plugin.get());
      if (candidate) { laser = candidate; ++count; }
    }
    const bool last = !plugins->empty() && plugins->back().get() == this;
    limo_fresh_layer::FreshGrid snapshot;
    snapshot.header.stamp = now;
    snapshot.header.frame_id = layered_costmap_->getGlobalFrameID();
    snapshot.scan_fresh = last && count == 1 && laser && laser->scanFreshAt(now, &snapshot.scan_stamp);
    if (!snapshot.scan_fresh) snapshot.scan_stamp = ros::Time(0);
    snapshot.grid.header = snapshot.header;
    snapshot.grid.info.map_load_time = now;
    snapshot.grid.info.resolution = master.getResolution();
    snapshot.grid.info.width = master.getSizeInCellsX();
    snapshot.grid.info.height = master.getSizeInCellsY();
    snapshot.grid.info.origin.position.x = master.getOriginX();
    snapshot.grid.info.origin.position.y = master.getOriginY();
    snapshot.grid.info.origin.orientation.w = 1.0;
    const size_t length = static_cast<size_t>(snapshot.grid.info.width) * snapshot.grid.info.height;
    const unsigned char* cells = master.getCharMap();
    snapshot.grid.data.resize(length);
    for (size_t i = 0; i < length; ++i) {
      const unsigned char value = cells[i];
      snapshot.grid.data[i] = value == 255 ? -1 : value == 254 ? 100 : value == 253 ? 99 :
                              value == 0 ? 0 : 1 + (97 * (value - 1)) / 251;
    }
    publisher_.publish(snapshot);
  }

 private:
  ros::Publisher publisher_;
};

}  // namespace limo_fresh_layer

PLUGINLIB_EXPORT_CLASS(limo_fresh_layer::FreshLaserLayer, costmap_2d::Layer)
PLUGINLIB_EXPORT_CLASS(limo_fresh_layer::FreshGridSnapshotLayer, costmap_2d::Layer)
