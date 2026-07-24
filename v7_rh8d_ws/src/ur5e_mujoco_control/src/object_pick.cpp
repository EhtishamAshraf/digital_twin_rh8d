/*
                                RH8D Hand Service Demo

  This ROS2 node only connects to the RH8D hand service and sends a grasp command. 
  This code can be used to test the behaviour of the hand by setting the initial position of the arm to the *cylinder pose* or *ball pose* from "initial_positions.yaml" file.

*/

#include <memory>
#include <chrono>
#include <string>
#include <thread>

#include <rclcpp/rclcpp.hpp>
#include <rh8d_mujoco_interfaces/srv/hand_command.hpp>

using HandCommand = rh8d_mujoco_interfaces::srv::HandCommand;

static const rclcpp::Logger LOGGER = rclcpp::get_logger("rh8d_hand_demo");

class RH8DHandDemo : public rclcpp::Node
{
public:
  RH8DHandDemo()
  : Node("rh8d_hand_demo")
  {
    hand_client_ =
        this->create_client<HandCommand>("/rh8d/hand_command");
  }

  void run()
  {
    RCLCPP_INFO(LOGGER, "Waiting for hand service...");

    rclcpp::sleep_for(std::chrono::seconds(10));
    // Open hand
    if (!call_hand_service("object", "open")) {
      RCLCPP_ERROR(LOGGER, "Failed to open hand");
      return;
    }

    rclcpp::sleep_for(std::chrono::seconds(5));

    // Grasp object
    RCLCPP_INFO(LOGGER, "Sending grasp command...");

    if (!call_hand_service("object", "grasp")) {
      RCLCPP_ERROR(LOGGER, "Failed to grasp object");
      return;
    }

    RCLCPP_INFO(LOGGER, "Grasp completed successfully.");
  }

private:
  bool call_hand_service(const std::string &mode, const std::string &gesture_name)
  {
    if (!hand_client_->wait_for_service(std::chrono::seconds(5))) {
      RCLCPP_ERROR(LOGGER, "Hand service /rh8d/hand_command not available");
      return false;
    }

    auto request = std::make_shared<HandCommand::Request>();

    request->mode = mode;
    request->gesture_name = gesture_name;

    auto future = hand_client_->async_send_request(request);

    auto status = future.wait_for(std::chrono::seconds(40));

    if (status != std::future_status::ready) {
      RCLCPP_ERROR(LOGGER, "Timed out waiting for hand service response");
      return false;
    }

    auto response = future.get();

    RCLCPP_INFO(LOGGER, "Hand service response: success=%s, message=%s", response->success ? "true" : "false", response->message.c_str());

    return response->success;
  }

  rclcpp::Client<HandCommand>::SharedPtr hand_client_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<RH8DHandDemo>();

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  std::thread spinner([&executor]() {
    executor.spin();
  });

  rclcpp::sleep_for(std::chrono::seconds(1));

  node->run();

  rclcpp::shutdown();
  spinner.join();

  return 0;
}