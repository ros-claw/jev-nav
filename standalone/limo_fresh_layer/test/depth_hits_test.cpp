// Exercise the production projector without a ROS master or hardware topics.
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
#include <sensor_msgs/point_cloud2_iterator.h>
#include <limo_fresh_layer/FreshGrid.h>
#include <gtest/gtest.h>

// This test supplies the same observations the inherited buffer supplies;
// it needs access to the stored observation but exercises real depthHits().
#define private public
#include "../src/fresh_laser_layer.cpp"
#undef private

class TestDepthLayer : public limo_fresh_layer::FreshDepthLayer {
 public:
  TestDepthLayer() {
    dsrv_ = nullptr;
    max_obstacle_height_ = 0.4;
    resizeMap(20, 20, 0.05, 0., 0.);
    setCost(2, 2, costmap_2d::LETHAL_OBSTACLE);
    setCost(6, 2, costmap_2d::LETHAL_OBSTACLE);
  }
};

static costmap_2d::Observation makeObservation(float x, float y, float z, ros::Time stamp) {
  sensor_msgs::PointCloud2 cloud;
  cloud.header.stamp = stamp;
  sensor_msgs::PointCloud2Modifier modifier(cloud);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(1);
  *sensor_msgs::PointCloud2Iterator<float>(cloud, "x") = x;
  *sensor_msgs::PointCloud2Iterator<float>(cloud, "y") = y;
  *sensor_msgs::PointCloud2Iterator<float>(cloud, "z") = z;
  costmap_2d::Observation observation(cloud, 3.);
  observation.origin_.x = observation.origin_.y = observation.origin_.z = 0.;
  return observation;
}

TEST(DepthHits, SameStampDifferentCloudReprojectsEvidence) {
  TestDepthLayer layer;
  costmap_2d::Costmap2D master(20, 20, 0.05, 0., 0., costmap_2d::NO_INFORMATION);
  const ros::Time stamp(10, 0);
  layer.depth_observation_.push_back(makeObservation(.125f, .125f, .1f, stamp));
  const auto first = layer.depthHits(master, stamp);
  ASSERT_EQ(1u, first.size());
  EXPECT_EQ(42u, first[0].index);

  layer.depth_observation_.clear();
  layer.depth_observation_.push_back(makeObservation(.325f, .125f, .2f, stamp));
  const auto second = layer.depthHits(master, stamp);
  ASSERT_EQ(1u, second.size());
  EXPECT_EQ(46u, second[0].index);
  EXPECT_FLOAT_EQ(.2f, second[0].min_z);
}

TEST(DepthHits, SameCloudUsesCurrentDepthLayerMarks) {
  TestDepthLayer layer;
  costmap_2d::Costmap2D master(20, 20, 0.05, 0., 0., costmap_2d::NO_INFORMATION);
  const ros::Time stamp(10, 0);
  layer.depth_observation_.push_back(makeObservation(.125f, .125f, .1f, stamp));
  ASSERT_EQ(1u, layer.depthHits(master, stamp).size());
  layer.setCost(2, 2, costmap_2d::NO_INFORMATION);
  EXPECT_TRUE(layer.depthHits(master, stamp).empty());
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
