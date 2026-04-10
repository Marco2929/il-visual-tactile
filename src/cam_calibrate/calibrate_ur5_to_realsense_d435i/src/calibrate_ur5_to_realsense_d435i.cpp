#include <rclcpp/rclcpp.hpp>
#include <ur_msgs/msg/io_states.hpp>
#include <ur_msgs/srv/set_io.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <thread>
#include <fstream>
#include <iomanip>
#include <ctime>
#include <filesystem>
#include <mutex>
#include <atomic>

using namespace std::chrono_literals;
namespace fs = std::filesystem;

class CalibrateUR5toRealSenseD435i : public rclcpp::Node
{
public:
  CalibrateUR5toRealSenseD435i() : Node("calibrate_ur5_to_realsense_d435i")
  {
    // Parameter mit Standardwerten
    this->declare_parameter<std::string>("pose_output_file", "tcp_poses.txt");
    this->declare_parameter<std::string>("image_output_folder", "images");
    
    this->get_parameter("pose_output_file", pose_output_file_);
    this->get_parameter("image_output_folder", image_output_folder_);
    
    // Erstelle Bildordner falls nicht vorhanden
    if (!fs::exists(image_output_folder_)) {
      fs::create_directories(image_output_folder_);
      RCLCPP_INFO(this->get_logger(), "Erstelle Bildordner: %s", image_output_folder_.c_str());
    }
    
    // Subscriptions
    io_subscription_ = this->create_subscription<ur_msgs::msg::IOStates>(
      "/io_and_status_controller/io_states", 10,
      std::bind(&CalibrateUR5toRealSenseD435i::io_callback, this, std::placeholders::_1));
    
    pose_subscription_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/tcp_pose_broadcaster/pose", 10,
      std::bind(&CalibrateUR5toRealSenseD435i::pose_callback, this, std::placeholders::_1));
    
    image_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/camera/color/image_raw", 10,
      std::bind(&CalibrateUR5toRealSenseD435i::image_callback, this, std::placeholders::_1));
    
    // Service Client
    client_ = this->create_client<ur_msgs::srv::SetIO>("/io_and_status_controller/set_io");
    
    RCLCPP_INFO(this->get_logger(), "Node gestartet - Überwache Digital Output 0");
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(pose_mutex_);
    current_tcp_pose_ = msg;
    pose_received_ = true;
  }
  
  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    current_image_ = msg;
    image_received_ = true;
  }

  void io_callback(const ur_msgs::msg::IOStates::SharedPtr msg)
  {
    if (signal_detected_.load()) {
      return;  // Sequenz läuft bereits
    }
    
    // Prüfe Digital Output 0
    for (const auto& digital_out : msg->digital_out_states)
    {
      if (digital_out.pin == 0 && digital_out.state == true)
      {
        bool expected = false;
        if (!signal_detected_.compare_exchange_strong(expected, true)) {
          return;  // Bereits getriggert
        }
        
        RCLCPP_INFO(this->get_logger(), "Digital Output 0 auf HIGH erkannt!");
        
        // Starte Verarbeitungssequenz in separatem Thread
        std::thread([this]() {
          // Schritt 1: Setze Digital Output 0 auf LOW
          set_digital_output(0, false);
          
          // Schritt 2: Speichere TCP-Pose und Bild
          save_pose_and_image();
          
          // Schritt 3: Setze Digital Output 1 auf HIGH
          set_digital_output(1, true);
          
          // Bereit für nächsten Trigger
          signal_detected_.store(false);
          RCLCPP_INFO(this->get_logger(), "Bereit für nächste Detektion");
          
        }).detach();
        
        break;
      }
    }
  }
  
  std::string get_timestamp_string()
  {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
  }
  
  std::string get_filename_timestamp()
  {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setfill('0') << std::setw(3) << now_ms.count();
    return ss.str();
  }
  
  std::string save_camera_image()
  {
    std::lock_guard<std::mutex> lock(image_mutex_);
    if (!image_received_) {
      RCLCPP_WARN(this->get_logger(), "Noch kein Kamerabild empfangen!");
      return "";
    }
    
    try {
      cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
        current_image_, sensor_msgs::image_encodings::BGR8);
      
      std::string filename = get_filename_timestamp() + ".png";
      std::string filepath = image_output_folder_ + "/" + filename;
      
      cv::imwrite(filepath, cv_ptr->image);
      RCLCPP_INFO(this->get_logger(), "Kamerabild gespeichert: %s", filepath.c_str());
      
      return filepath;
      
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Fehler beim Speichern des Bildes: %s", e.what());
      return "";
    }
  }
  
  void save_pose_and_image()
  {
    geometry_msgs::msg::PoseStamped::SharedPtr pose;
    
    {
      std::lock_guard<std::mutex> lock(pose_mutex_);
      if (!pose_received_) {
        RCLCPP_WARN(this->get_logger(), "Noch keine TCP-Pose empfangen!");
        return;
      }
      pose = current_tcp_pose_;
    }
    
    std::string timestamp = get_timestamp_string();
    std::string image_path = save_camera_image();
    
    // Speichere in Datei
    std::ofstream outfile(pose_output_file_, std::ios_base::app);
    
    if (outfile.is_open()) {
      outfile << "Zeitstempel: " << timestamp << "\n";
      outfile << "Position: "
              << pose->pose.position.x << ", "
              << pose->pose.position.y << ", "
              << pose->pose.position.z << "\n";
      outfile << "Orientierung (Quaternion): "
              << pose->pose.orientation.x << ", "
              << pose->pose.orientation.y << ", "
              << pose->pose.orientation.z << ", "
              << pose->pose.orientation.w << "\n";
      outfile << "Bild: " << (image_path.empty() ? "FEHLER" : image_path) << "\n";
      outfile << "========================================\n";
      outfile.close();
      
      RCLCPP_INFO(this->get_logger(), "TCP-Pose und Bildpfad gespeichert");
    } else {
      RCLCPP_ERROR(this->get_logger(), "Fehler beim Öffnen der Datei '%s'", pose_output_file_.c_str());
    }
  }
  
  bool set_digital_output(int pin, bool state)
  {
    if (!client_->wait_for_service(3s)) {
      RCLCPP_ERROR(this->get_logger(), "SetIO Service nicht verfügbar!");
      return false;
    }
    
    auto request = std::make_shared<ur_msgs::srv::SetIO::Request>();
    request->fun = 1;  // Digital Output
    request->pin = pin;
    request->state = state ? 1.0 : 0.0;
    
    RCLCPP_INFO(this->get_logger(), "Setze Digital Output %d auf %s", 
                pin, state ? "HIGH" : "LOW");
    
    auto result_future = client_->async_send_request(request);
    
    if (result_future.wait_for(5s) != std::future_status::ready) {
      RCLCPP_ERROR(this->get_logger(), "Timeout beim SetIO Service");
      return false;
    }
    
    try {
      auto result = result_future.get();
      if (!result->success) {
        RCLCPP_ERROR(this->get_logger(), "SetIO Service meldet Fehler");
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception beim Service-Aufruf: %s", e.what());
      return false;
    }
  }
  
  // Subscriptions und Service Client
  rclcpp::Subscription<ur_msgs::msg::IOStates>::SharedPtr io_subscription_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Client<ur_msgs::srv::SetIO>::SharedPtr client_;
  
  // Daten
  geometry_msgs::msg::PoseStamped::SharedPtr current_tcp_pose_;
  sensor_msgs::msg::Image::SharedPtr current_image_;
  bool pose_received_ = false;
  bool image_received_ = false;
  
  // Thread-Synchronisation
  std::mutex pose_mutex_;
  std::mutex image_mutex_;
  std::atomic<bool> signal_detected_{false};
  
  // Parameter
  std::string pose_output_file_;
  std::string image_output_folder_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  
  try {
    auto node = std::make_shared<CalibrateUR5toRealSenseD435i>();
    rclcpp::spin(node);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("main"), "Exception: %s", e.what());
  }
  
  rclcpp::shutdown();
  return 0;
}