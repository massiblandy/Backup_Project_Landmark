/**
 *
 *  \file
 *  \brief      Main entry point for UM7 driver. Handles serial connection
 *              details, as well as all ROS message stuffing, parameters,
 *              topics, etc.
 *  \author     Mike Purvis <mpurvis@clearpathrobotics.com> (original code for UM6)
 *  \copyright  Copyright (c) 2013, Clearpath Robotics, Inc.
 *  \author     Alex Brown <rbirac@cox.net>		    (adapted to UM7)
 *  \copyright  Copyright (c) 2015, Alex Brown.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Clearpath Robotics, Inc. nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL CLEARPATH ROBOTICS, INC. OR ALEX BROWN BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// ros2 service call /imu/reset custom_interfaces/srv/Reset "{reset_ekf: True}"

#include <string>
#include <math.h> 
#include "um7/um7.h"
#include "custom_interfaces/srv/reset.hpp"

const char VERSION[10] = "0.0.2";   // um7_driver version

// Don't try to be too clever. Arrival of this message triggers
// us to publish everything we have.
const uint8_t TRIGGER_PACKET = DREG_EULER_PHI_THETA;

namespace um7
{

/**
 * Function generalizes the process of commanding the UM7 via one of its command
 * registers.
 */
template<typename RegT>
void Driver::send_command(
  const um7::Accessor<RegT>& reg,
  std::string human_name)
{
  RCLCPP_INFO_STREAM(this->get_logger(), "Sending command: " << human_name);
  if (!sensor_->sendWaitAck(reg))
  {
    throw std::runtime_error("Command to device failed.");
  }
}

void Driver::handle_reset_service(
  const std::shared_ptr<custom_interfaces::srv::Reset::Request> req,
  std::shared_ptr<custom_interfaces::srv::Reset::Response> resp)
{
  um7::Registers r;
  if (req->zero_gyros) send_command(r.cmd_zero_gyros, "zero gyroscopes");
  if (req->reset_ekf) send_command(r.cmd_reset_ekf, "reset EKF");     
  if (req->set_mag_ref) send_command(r.cmd_set_mag_ref, "set magnetometer reference");
}

/**
 * Send configuration messages to the UM7, critically, to turn on the value outputs
 * which we require, and inject necessary configuration parameters.
 */
void Driver::configure_sensor(std::shared_ptr<um7::Comms> sensor)
{
  um7::Registers r;

  uint32_t comm_reg = (BAUD_115200 << COM_BAUD_START);
  r.communication.set(0, comm_reg);
  if (!sensor->sendWaitAck(r.comrate2))
  {
    throw std::runtime_error("Unable to set CREG_COM_SETTINGS.");
  }

  // set the broadcast rate of the device
  int rate;
  this->get_parameter("update_rate", rate);
  if (rate < 20 || rate > 255)
  {
    RCLCPP_WARN(this->get_logger(), "Potentially unsupported update rate of %d", rate);
  }

  uint32_t rate_bits = static_cast<uint32_t>(rate);
  RCLCPP_INFO(this->get_logger(), "Setting update rate to %uHz", rate);
  uint32_t raw_rate = (rate_bits << RATE2_ALL_RAW_START);
  r.comrate2.set(0, raw_rate);
  if (!sensor->sendWaitAck(r.comrate2))
  {
    throw std::runtime_error("Unable to set CREG_COM_RATES2.");
  }

  uint32_t proc_rate = (rate_bits << RATE4_ALL_PROC_START);
  r.comrate4.set(0, proc_rate);
  if (!sensor->sendWaitAck(r.comrate4))
  {
    throw std::runtime_error("Unable to set CREG_COM_RATES4.");
  }

  uint32_t misc_rate = (rate_bits << RATE5_EULER_START) | (rate_bits << RATE5_QUAT_START);
  r.comrate5.set(0, misc_rate);
  if (!sensor->sendWaitAck(r.comrate5))
  {
    throw std::runtime_error("Unable to set CREG_COM_RATES5.");
  }

  uint32_t health_rate = (5 << RATE6_HEALTH_START);  // note:  5 gives 2 hz rate
  r.comrate6.set(0, health_rate);
  if (!sensor->sendWaitAck(r.comrate6))
  {
    throw std::runtime_error("Unable to set CREG_COM_RATES6.");
  }

  // Options available using parameters)
  uint32_t misc_config_reg = 0;  // initialize all options off

  // Optionally disable mag updates in the sensor's EKF.
  bool mag_updates;
  this->get_parameter("mag_updates", mag_updates);
  if (mag_updates)
  {
    misc_config_reg |= MAG_UPDATES_ENABLED;
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Excluding magnetometer updates from EKF.");
  }

  // Optionally enable quaternion mode .
  bool quat_mode;
  this->get_parameter("quat_mode", quat_mode);
  if (quat_mode)
  {
    misc_config_reg |= QUATERNION_MODE_ENABLED;
  }
  else
  {
    RCLCPP_WARN(this->get_logger(), "Excluding quaternion mode.");
  }

  r.misc_config.set(0, misc_config_reg);
  if (!sensor_->sendWaitAck(r.misc_config))
  {
    throw std::runtime_error("Unable to set CREG_MISC_SETTINGS.");
  }

  // Optionally disable performing a zero gyros command on driver startup.
  bool zero_gyros;
  this->get_parameter("zero_gyros", zero_gyros);
  if (zero_gyros)
  {
    send_command(r.cmd_zero_gyros, "zero gyroscopes");
  }
}

/**
 * Uses the register accessors to grab data from the IMU, and populate
 * the ROS messages which are output.
 */
void Driver::publish(um7::Registers& r)
{
  
  double new_value[3] = {0,0,0}; // x, y, z
  
  if (imu_pub_->get_subscription_count() > 0)
  {
    switch (axes_)
    {
      case OutputAxisOptions::ENU:
      {
        // body-fixed frame NED to ENU: (x y z)->(x -y -z) or (w x y z)->(x -y -z w)
        // world frame      NED to ENU: (x y z)->(y  x -z) or (w x y z)->(y  x -z w)
        // world frame
        imu_msg_.orientation.w =  r.quat.get_scaled(2);
        imu_msg_.orientation.x =  r.quat.get_scaled(1);
        imu_msg_.orientation.y = -r.quat.get_scaled(3);
        imu_msg_.orientation.z =  r.quat.get_scaled(0);

        // body-fixed frame
        imu_msg_.angular_velocity.x =  r.gyro.get_scaled(0);
        imu_msg_.angular_velocity.y = -r.gyro.get_scaled(1);
        imu_msg_.angular_velocity.z = -r.gyro.get_scaled(2);

        // body-fixed frame
        imu_msg_.linear_acceleration.x =  r.accel.get_scaled(0);
        imu_msg_.linear_acceleration.y = -r.accel.get_scaled(1);
        imu_msg_.linear_acceleration.z = -r.accel.get_scaled(2);
        break;
      }
      case OutputAxisOptions::ROBOT_FRAME:
      {
        // body-fixed frame
        imu_msg_.orientation.w = -r.quat.get_scaled(0);
        imu_msg_.orientation.x = -r.quat.get_scaled(1);
        imu_msg_.orientation.y =  r.quat.get_scaled(2);
        imu_msg_.orientation.z =  r.quat.get_scaled(3);

        // body-fixed frame
        imu_msg_.angular_velocity.x =  r.gyro.get_scaled(0);
        imu_msg_.angular_velocity.y = -r.gyro.get_scaled(1);
        imu_msg_.angular_velocity.z = -r.gyro.get_scaled(2);

        // body-fixed frame
        imu_msg_.linear_acceleration.x =  r.accel.get_scaled(0);
        imu_msg_.linear_acceleration.y = -r.accel.get_scaled(1);
        imu_msg_.linear_acceleration.z = -r.accel.get_scaled(2);
        break;
      }
      case OutputAxisOptions::DEFAULT:
      {
        imu_msg_.orientation.w = r.quat.get_scaled(0);
        imu_msg_.orientation.x = r.quat.get_scaled(1);
        imu_msg_.orientation.y = r.quat.get_scaled(2);
        imu_msg_.orientation.z = r.quat.get_scaled(3);

        imu_msg_.angular_velocity.x = r.gyro.get_scaled(0);
        imu_msg_.angular_velocity.y = r.gyro.get_scaled(1);
        imu_msg_.angular_velocity.z = r.gyro.get_scaled(2);

        imu_msg_.linear_acceleration.x = r.accel.get_scaled(0);
        imu_msg_.linear_acceleration.y = r.accel.get_scaled(1);
        imu_msg_.linear_acceleration.z = r.accel.get_scaled(2);
        break;
      }
      default:
        RCLCPP_ERROR(this->get_logger(), "OuputAxes enum value invalid");
    }

    imu_pub_->publish(imu_msg_);
  }

  // Magnetometer.  transform to ROS axes
  if (mag_pub_->get_subscription_count() > 0)
  {
    sensor_msgs::msg::MagneticField mag_msg;
    mag_msg.header = imu_msg_.header;

    switch (axes_)
    {
      case OutputAxisOptions::ENU:
      {
        mag_msg.magnetic_field.x = r.mag.get_scaled(1);
        mag_msg.magnetic_field.y = r.mag.get_scaled(0);
        mag_msg.magnetic_field.z = -r.mag.get_scaled(2);
        break;
      }
      case OutputAxisOptions::ROBOT_FRAME:
      {
        // body-fixed frame
        mag_msg.magnetic_field.x =  r.mag.get_scaled(0);
        mag_msg.magnetic_field.y = -r.mag.get_scaled(1);
        mag_msg.magnetic_field.z = -r.mag.get_scaled(2);
        break;
      }
      case OutputAxisOptions::DEFAULT:
      {
        mag_msg.magnetic_field.x = r.mag.get_scaled(0);
        mag_msg.magnetic_field.y = r.mag.get_scaled(1);
        mag_msg.magnetic_field.z = r.mag.get_scaled(2);
        break;
      }
      default:
        RCLCPP_ERROR(this->get_logger(), "OuputAxes enum value invalid");
    }

    mag_pub_->publish(mag_msg);
  }

  // Euler attitudes.  transform to ROS axes
  if (rpy_pub_->get_subscription_count() > 0)
  {
    geometry_msgs::msg::Vector3Stamped rpy_msg;
    rpy_msg.header = imu_msg_.header;

    if(this->first_offset)
    {
      if(this->cont_for_offset > 50)
      {
        for(int i=0; i<3; i++)
        {
          this->offset[i] = r.euler.get_scaled(i);
        }
        this->first_offset = false;        
      }
      this->cont_for_offset++;
    }

    for(int i=0; i<3; i++)
    {
      new_value[i] = r.euler.get_scaled(i) - this->offset[i];
      if(new_value[i]<-3.14) new_value[i] = 3.14 - fmod(fabs(new_value[i]), 3.14);
      else if (new_value[i] > 3.14) new_value[i] =  -3.14 + fmod(fabs(new_value[i]), 3.14);
      
    }

    switch (axes_)
    {
      case OutputAxisOptions::ENU:
      {
        // world frame
        rpy_msg.vector.x = new_value[0];
        rpy_msg.vector.y = new_value[1];
        rpy_msg.vector.z = -new_value[2];
        break;
      }
      case OutputAxisOptions::ROBOT_FRAME:
      {
        rpy_msg.vector.x =  new_value[0];
        rpy_msg.vector.y = -new_value[1];
        rpy_msg.vector.z = -new_value[2];
        break;
      }
      case OutputAxisOptions::DEFAULT:
      {
        rpy_msg.vector.x = new_value[0];
        rpy_msg.vector.y = new_value[1];
        rpy_msg.vector.z = new_value[2];
        
        break;
      }
      default:
        RCLCPP_ERROR(this->get_logger(), "OuputAxes enum value invalid");
    }



    rpy_pub_->publish(rpy_msg);
  }

  // Temperature
  if (temperature_pub_->get_subscription_count() > 0)
  {
    std_msgs::msg::Float32 temp_msg;
    temp_msg.data = r.temperature.get_scaled(0);
    temperature_pub_->publish(temp_msg);
  }
}

Driver::Driver(const rclcpp::NodeOptions & options) :
  rclcpp::Node("um7_driver", options),
  axes_(OutputAxisOptions::DEFAULT)
{
  // Load parameters
  port_ = this->declare_parameter<std::string>("port", "/dev/ttyUSB1");
  int32_t baud = this->declare_parameter<int32_t>("baud", 115200);

  serial_.setPort(port_);
  serial_.setBaudrate(baud);
  serial::Timeout to = serial::Timeout(50, 50, 0, 50, 0);
  serial_.setTimeout(to);

  imu_msg_.header.frame_id = this->declare_parameter<std::string>("frame_id", "imu_link");
  // Defaults obtained experimentally from hardware, no device spec exists
  double linear_acceleration_stdev = this->declare_parameter<double>("linear_acceleration_stdev", (4.0 * 1e-3f * 9.80665));
  double angular_velocity_stdev = this->declare_parameter<double>("angular_velocity_stdev", (0.06 * 3.14159 / 180.0));

  double linear_acceleration_cov = linear_acceleration_stdev * linear_acceleration_stdev;
  double angular_velocity_cov = angular_velocity_stdev * angular_velocity_stdev;

  // From the UM7 datasheet for the dynamic accuracy from the EKF.
  double orientation_x_stdev = this->declare_parameter<double>("orientation_x_stdev", (3.0 * 3.14159 / 180.0));
  double orientation_y_stdev = this->declare_parameter<double>("orientation_y_stdev", (3.0 * 3.14159 / 180.0));
  double orientation_z_stdev = this->declare_parameter<double>("orientation_z_stdev", (3.0 * 3.14159 / 180.0));

  double orientation_x_covar = orientation_x_stdev * orientation_x_stdev;
  double orientation_y_covar = orientation_y_stdev * orientation_y_stdev;
  double orientation_z_covar = orientation_z_stdev * orientation_z_stdev;

  // Enable converting from NED to ENU by default
  bool tf_ned_to_enu = this->declare_parameter<bool>("tf_ned_to_enu", true);
  bool orientation_in_robot_frame = this->declare_parameter<bool>("orientation_in_robot_frame", false);
  if (tf_ned_to_enu && orientation_in_robot_frame)
  {
    RCLCPP_ERROR(this->get_logger(), "Requested IMU data in two separate frames.");
  }
  else if (tf_ned_to_enu)
  {
    axes_ = OutputAxisOptions::ENU;
  }
  else if (orientation_in_robot_frame)
  {
    axes_ = OutputAxisOptions::ROBOT_FRAME;
  }

  // Parameters for configure_sensor
  this->declare_parameter<int>("update_rate", 20);
  this->declare_parameter<bool>("mag_updates", false);
  this->declare_parameter<bool>("quat_mode", true);
  this->declare_parameter<bool>("zero_gyros", true);

  // These values do not need to be converted
  imu_msg_.linear_acceleration_covariance[0] = linear_acceleration_cov;
  imu_msg_.linear_acceleration_covariance[4] = linear_acceleration_cov;
  imu_msg_.linear_acceleration_covariance[8] = linear_acceleration_cov;

  imu_msg_.angular_velocity_covariance[0] = angular_velocity_cov;
  imu_msg_.angular_velocity_covariance[4] = angular_velocity_cov;
  imu_msg_.angular_velocity_covariance[8] = angular_velocity_cov;

  imu_msg_.orientation_covariance[0] = orientation_x_covar;
  imu_msg_.orientation_covariance[4] = orientation_y_covar;
  imu_msg_.orientation_covariance[8] = orientation_z_covar;

  // Create ROS interfaces
  imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data", 1);
  mag_pub_ = this->create_publisher<sensor_msgs::msg::MagneticField>("imu/mag", 1);
  rpy_pub_ = this->create_publisher<geometry_msgs::msg::Vector3Stamped>("imu/rpy", 1);
  temperature_pub_ = this->create_publisher<std_msgs::msg::Float32>("imu/temperature", 1);

  // Start real time thread
  update_thread_ = std::thread(std::bind(&Driver::update_loop, this));
}

void Driver::update_loop(void)
{
  // Real Time Loop
  bool first_failure = true;
  while (rclcpp::ok())
  {
    try
    {
      serial_.open();
    }
    catch (const serial::IOException& e)
    {
      RCLCPP_WARN(this->get_logger(), "um7_driver was unable to connect to port %s.", port_.c_str());
    }
    if (serial_.isOpen())
    {
      RCLCPP_INFO(this->get_logger(), "um7_driver successfully connected to serial port %s.", port_.c_str());
      first_failure = true;
      try
      {
        sensor_.reset(new um7::Comms(&serial_));
        configure_sensor(sensor_);
        um7::Registers registers;
        auto service = this->create_service<custom_interfaces::srv::Reset>("imu/reset",
          std::bind(&Driver::handle_reset_service, this, std::placeholders::_1, std::placeholders::_2));

        while (rclcpp::ok())
        {
          // triggered by arrival of last message packet
          if (sensor_->receive(&registers) == TRIGGER_PACKET)
          {
            // Triggered by arrival of final message in group.
            imu_msg_.header.stamp = this->now();
            this->publish(registers);
          }
        }
      }
      catch(const std::exception& e)
      {
        if (serial_.isOpen())
        {
          serial_.close();
        }
        RCLCPP_ERROR_STREAM(this->get_logger(), e.what());
        RCLCPP_INFO(this->get_logger(), "Attempting reconnection after error.");
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
    }
    else
    {
      RCLCPP_WARN_STREAM_EXPRESSION(this->get_logger(),
        first_failure,
        "Could not connect to serial device " << port_ << ". Trying again every 1 second.");
      first_failure = false;
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
}

}  // namespace um7

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(um7::Driver)
