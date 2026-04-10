#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco.hpp>
#include <deque>

class CharucoDetectorNode : public rclcpp::Node
{
public:
    CharucoDetectorNode() : Node("charuco_detector_node"), camera_calibrated_(false)
    {
        // Board parameters
        SQUARES_X = 14;
        SQUARES_Y = 9;
        SQUARE_LENGTH = 0.020f;  // 20mm in meters
        MARKER_LENGTH = 0.015f;  // 15mm in meters
/**  
        SQUARES_X = 12;
        SQUARES_Y = 9;
        SQUARE_LENGTH = 0.015f;  // 15mm in meters
        MARKER_LENGTH = 0.011f;  // 11mm in meters
*/
        // Filtering parameters
        MIN_CORNERS_THRESHOLD = 10;  // Minimum corners for valid detection
        SMOOTHING_WINDOW = 5;        // Number of frames to average
        
        // Initialize ArUco dictionary and board
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_250);
        board_ = cv::aruco::CharucoBoard::create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH,
            MARKER_LENGTH,
            aruco_dict_
        );
        
        // Improved detector parameters for stability
        parameters_ = cv::aruco::DetectorParameters::create();
        parameters_->adaptiveThreshWinSizeMin = 3;
        parameters_->adaptiveThreshWinSizeMax = 23;
        parameters_->adaptiveThreshWinSizeStep = 10;
        parameters_->adaptiveThreshConstant = 7;
        parameters_->minMarkerPerimeterRate = 0.03;
        parameters_->maxMarkerPerimeterRate = 4.0;
        parameters_->polygonalApproxAccuracyRate = 0.03;
        parameters_->minCornerDistanceRate = 0.05;
        parameters_->minDistanceToBorder = 3;
        parameters_->minMarkerDistanceRate = 0.05;
        parameters_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        parameters_->cornerRefinementWinSize = 5;
        parameters_->cornerRefinementMaxIterations = 30;
        parameters_->cornerRefinementMinAccuracy = 0.1;

        // Initialize TF broadcaster
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

        // Create subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color/image_raw",
            10,
            std::bind(&CharucoDetectorNode::imageCallback, this, std::placeholders::_1)
        );

        camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/color/camera_info",
            10,
            std::bind(&CharucoDetectorNode::cameraInfoCallback, this, std::placeholders::_1)
        );

        // Create publisher
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>(
            "/charuco/debug_image",
            10
        );

        RCLCPP_INFO(this->get_logger(), "ChArUco Detector initialized (Stabilized)");
        RCLCPP_INFO(this->get_logger(), "Board: %dx%d, Square: %.1fmm, Marker: %.1fmm", 
                    SQUARES_X, SQUARES_Y, SQUARE_LENGTH*1000, MARKER_LENGTH*1000);
        RCLCPP_INFO(this->get_logger(), "Min corners threshold: %d, Smoothing window: %d", 
                    MIN_CORNERS_THRESHOLD, SMOOTHING_WINDOW);
        RCLCPP_INFO(this->get_logger(), "Subscribing to:");
        RCLCPP_INFO(this->get_logger(), "  - /camera/color/image_raw");
        RCLCPP_INFO(this->get_logger(), "  - /camera/color/camera_info");
    }

private:
    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (!camera_calibrated_)
        {
            // Extract camera matrix
            camera_matrix_ = cv::Mat(3, 3, CV_64F);
            for (int i = 0; i < 9; i++)
            {
                camera_matrix_.at<double>(i / 3, i % 3) = msg->k[i];
            }

            // Extract distortion coefficients
            dist_coeffs_ = cv::Mat(msg->d.size(), 1, CV_64F);
            for (size_t i = 0; i < msg->d.size(); i++)
            {
                dist_coeffs_.at<double>(i) = msg->d[i];
            }

            camera_calibrated_ = true;
            RCLCPP_INFO(this->get_logger(), "Camera calibration loaded");
        }
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!camera_calibrated_)
        {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "Waiting for camera calibration...");
            return;
        }

        // Convert ROS Image to OpenCV
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        // Optional: Apply Gaussian blur to reduce noise
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        // Detect ArUco markers
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        
        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, parameters_, rejected);

        // Create debug image
        cv::Mat debug_image = cv_ptr->image.clone();

        if (!ids.empty() && ids.size() > 4)
        {
            // Draw detected markers
            cv::aruco::drawDetectedMarkers(debug_image, corners, ids);

            // Interpolate ChArUco corners
            std::vector<cv::Point2f> charuco_corners;
            std::vector<int> charuco_ids;
            
            int interpolated = cv::aruco::interpolateCornersCharuco(
                corners, ids, gray, board_, charuco_corners, charuco_ids
            );

            // Only process if we have enough corners for stable detection
            if (interpolated >= MIN_CORNERS_THRESHOLD)
            {
                // Draw ChArUco corners
                cv::aruco::drawDetectedCornersCharuco(debug_image, charuco_corners, charuco_ids);

                // Estimate pose
                cv::Vec3d rvec, tvec;
                bool valid = cv::aruco::estimatePoseCharucoBoard(
                    charuco_corners,
                    charuco_ids,
                    board_,
                    camera_matrix_,
                    dist_coeffs_,
                    rvec,
                    tvec
                );

                if (valid)
                {
                    // Apply temporal smoothing
                    cv::Vec3d smoothed_rvec, smoothed_tvec;
                    if (applySmoothing(rvec, tvec, smoothed_rvec, smoothed_tvec))
                    {
                        // Draw axis
                        cv::drawFrameAxes(debug_image, camera_matrix_, dist_coeffs_, 
                                         smoothed_rvec, smoothed_tvec, 0.05f);

                        // Publish TF
                        publishTransform(smoothed_rvec, smoothed_tvec, msg->header);

                        // Log position
                        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "Board detected: %d corners | Pos: [%.3f, %.3f, %.3f]",
                            interpolated, smoothed_tvec[0], smoothed_tvec[1], smoothed_tvec[2]
                        );
                    }
                }
            }
            else
            {
                // Show warning on image
                cv::putText(debug_image, 
                    "Too few corners: " + std::to_string(interpolated) + " < " + std::to_string(MIN_CORNERS_THRESHOLD),
                    cv::Point(10, 30), 
                    cv::FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    cv::Scalar(0, 0, 255), 
                    2);
            }
        }

        // Publish debug image
        sensor_msgs::msg::Image::SharedPtr debug_msg = 
            cv_bridge::CvImage(msg->header, "bgr8", debug_image).toImageMsg();
        debug_pub_->publish(*debug_msg);
    }

    bool applySmoothing(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                       cv::Vec3d& smoothed_rvec, cv::Vec3d& smoothed_tvec)
    {
        // Add current measurement to history
        PoseData current;
        current.rvec = rvec;
        current.tvec = tvec;
        pose_history_.push_back(current);

        // Keep only recent history
        if (pose_history_.size() > SMOOTHING_WINDOW)
        {
            pose_history_.pop_front();
        }

        // Need at least 3 samples for good smoothing
        if (pose_history_.size() < 3)
        {
            smoothed_rvec = rvec;
            smoothed_tvec = tvec;
            return true;
        }

        // Calculate weighted average (more weight to recent samples)
        smoothed_rvec = cv::Vec3d(0, 0, 0);
        smoothed_tvec = cv::Vec3d(0, 0, 0);
        double total_weight = 0.0;

        for (size_t i = 0; i < pose_history_.size(); i++)
        {
            // Linear weighting: recent samples get more weight
            double weight = (i + 1.0) / pose_history_.size();
            total_weight += weight;
            
            smoothed_rvec += pose_history_[i].rvec * weight;
            smoothed_tvec += pose_history_[i].tvec * weight;
        }

        smoothed_rvec /= total_weight;
        smoothed_tvec /= total_weight;

        return true;
    }

    void publishTransform(const cv::Vec3d& rvec, const cv::Vec3d& tvec,
                         const std_msgs::msg::Header& header)
    {
        geometry_msgs::msg::TransformStamped t;
        t.header = header;
        t.header.frame_id = "camera_color_optical_frame";
        t.child_frame_id = "charuco_board";

        // Translation
        t.transform.translation.x = tvec[0];
        t.transform.translation.y = tvec[1];
        t.transform.translation.z = tvec[2];

        // Rotation (convert from rvec to quaternion)
        cv::Mat rotation_matrix;
        cv::Rodrigues(rvec, rotation_matrix);
        
        double qx, qy, qz, qw;
        rotationMatrixToQuaternion(rotation_matrix, qx, qy, qz, qw);
        
        t.transform.rotation.x = qx;
        t.transform.rotation.y = qy;
        t.transform.rotation.z = qz;
        t.transform.rotation.w = qw;

        tf_broadcaster_->sendTransform(t);
    }

    void rotationMatrixToQuaternion(const cv::Mat& R, 
                                    double& qx, double& qy, 
                                    double& qz, double& qw)
    {
        double trace = R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2);
        
        if (trace > 0.0)
        {
            double s = 0.5 / std::sqrt(trace + 1.0);
            qw = 0.25 / s;
            qx = (R.at<double>(2, 1) - R.at<double>(1, 2)) * s;
            qy = (R.at<double>(0, 2) - R.at<double>(2, 0)) * s;
            qz = (R.at<double>(1, 0) - R.at<double>(0, 1)) * s;
        }
        else
        {
            if (R.at<double>(0, 0) > R.at<double>(1, 1) && R.at<double>(0, 0) > R.at<double>(2, 2))
            {
                double s = 2.0 * std::sqrt(1.0 + R.at<double>(0, 0) - R.at<double>(1, 1) - R.at<double>(2, 2));
                qw = (R.at<double>(2, 1) - R.at<double>(1, 2)) / s;
                qx = 0.25 * s;
                qy = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
                qz = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
            }
            else if (R.at<double>(1, 1) > R.at<double>(2, 2))
            {
                double s = 2.0 * std::sqrt(1.0 + R.at<double>(1, 1) - R.at<double>(0, 0) - R.at<double>(2, 2));
                qw = (R.at<double>(0, 2) - R.at<double>(2, 0)) / s;
                qx = (R.at<double>(0, 1) + R.at<double>(1, 0)) / s;
                qy = 0.25 * s;
                qz = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
            }
            else
            {
                double s = 2.0 * std::sqrt(1.0 + R.at<double>(2, 2) - R.at<double>(0, 0) - R.at<double>(1, 1));
                qw = (R.at<double>(1, 0) - R.at<double>(0, 1)) / s;
                qx = (R.at<double>(0, 2) + R.at<double>(2, 0)) / s;
                qy = (R.at<double>(1, 2) + R.at<double>(2, 1)) / s;
                qz = 0.25 * s;
            }
        }
    }

    // Pose data structure for history
    struct PoseData
    {
        cv::Vec3d rvec;
        cv::Vec3d tvec;
    };

    // Board parameters
    int SQUARES_X;
    int SQUARES_Y;
    float SQUARE_LENGTH;
    float MARKER_LENGTH;
    
    // Filtering parameters
    int MIN_CORNERS_THRESHOLD;
    int SMOOTHING_WINDOW;

    // ArUco objects
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    cv::Ptr<cv::aruco::DetectorParameters> parameters_;

    // Camera calibration
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool camera_calibrated_;

    // Pose history for smoothing
    std::deque<PoseData> pose_history_;

    // ROS2 objects
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CharucoDetectorNode>());
    rclcpp::shutdown();
    return 0;
}