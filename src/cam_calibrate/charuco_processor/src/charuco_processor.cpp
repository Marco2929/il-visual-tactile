#include <opencv2/opencv.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>

namespace fs = std::filesystem;

struct RobotPose
{
    std::string timestamp;
    std::string image_name;
    cv::Vec3d position;      // x, y, z
    cv::Vec4d quaternion;    // qx, qy, qz, qw
    
    cv::Mat getRotationMatrix() const
    {
        // Convert quaternion to rotation matrix
        double qx = quaternion[0];
        double qy = quaternion[1];
        double qz = quaternion[2];
        double qw = quaternion[3];
        
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        
        R.at<double>(0, 0) = 1 - 2*qy*qy - 2*qz*qz;
        R.at<double>(0, 1) = 2*qx*qy - 2*qz*qw;
        R.at<double>(0, 2) = 2*qx*qz + 2*qy*qw;
        
        R.at<double>(1, 0) = 2*qx*qy + 2*qz*qw;
        R.at<double>(1, 1) = 1 - 2*qx*qx - 2*qz*qz;
        R.at<double>(1, 2) = 2*qy*qz - 2*qx*qw;
        
        R.at<double>(2, 0) = 2*qx*qz - 2*qy*qw;
        R.at<double>(2, 1) = 2*qy*qz + 2*qx*qw;
        R.at<double>(2, 2) = 1 - 2*qx*qx - 2*qy*qy;
        
        return R;
    }
    
    cv::Mat getTransformationMatrix() const
    {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R = getRotationMatrix();
        
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        T.at<double>(0, 3) = position[0];
        T.at<double>(1, 3) = position[1];
        T.at<double>(2, 3) = position[2];
        
        return T;
    }
};

struct DetectedPose
{
    std::string image_name;
    cv::Vec3d rvec;
    cv::Vec3d tvec;
    
    cv::Mat getRotationMatrix() const
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        return R;
    }
    
    cv::Mat getTransformationMatrix() const
    {
        cv::Mat T = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R = getRotationMatrix();
        
        R.copyTo(T(cv::Rect(0, 0, 3, 3)));
        T.at<double>(0, 3) = tvec[0];
        T.at<double>(1, 3) = tvec[1];
        T.at<double>(2, 3) = tvec[2];
        
        return T;
    }
};

struct MeasurementQuality
{
    std::string image_name;
    cv::Mat T_base2camera;
    double deviation_from_guess;
    double weight;
    bool is_outlier;
};

class CharucoProcessor
{
public:
    CharucoProcessor(const std::string& input_dir, 
                     const std::string& tcp_poses_file,
                     const std::string& calib_file = "",
                     const cv::Vec3d& initial_guess_position = cv::Vec3d(0, 0, 0),
                     bool use_initial_guess = false,
                     double outlier_threshold_meters = 0.1)
        : input_dir_(input_dir), 
          tcp_poses_file_(tcp_poses_file),
          camera_calibrated_(false),
          initial_guess_position_(initial_guess_position),
          use_initial_guess_(use_initial_guess),
          outlier_threshold_(outlier_threshold_meters)
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

        // Load camera calibration
        if (!calib_file.empty() && loadCameraCalibration(calib_file))
        {
            std::cout << "Camera calibration loaded from: " << calib_file << std::endl;
        }
        else
        {
            std::cout << "WARNING: No camera calibration loaded. Using default parameters." << std::endl;
            std::cout << "Pose estimation will not be accurate!" << std::endl;
            camera_matrix_ = (cv::Mat_<double>(3, 3) << 
                899.4387817382812,   0.0,                 650.6546020507812,
                0.0,                 899.4373779296875,   353.9114074707031,
                0.0,                 0.0,                 1.0);
            
            // FIX: Use the actual distortion coefficients from your camera_info
            // Order: k1, k2, p1, p2, k3
            dist_coeffs_ = (cv::Mat_<double>(5, 1) << 
                 0.1675209403038025,
                -0.49356043338775635,
                -0.0008838649373501539,
                 0.00035886315163224936,
                 0.4410543441772461);
            
            camera_calibrated_ = true;
        }

        // Load robot TCP poses
        if (!tcp_poses_file_.empty())
        {
            loadTCPPoses();
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "ChArUco Processor - EYE-TO-HAND Setup" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Setup: Camera FIXED, ChArUco Board on Robot TCP" << std::endl;
        std::cout << "Board: " << SQUARES_X << "x" << SQUARES_Y 
                  << ", Square: " << SQUARE_LENGTH*1000 << "mm"
                  << ", Marker: " << MARKER_LENGTH*1000 << "mm" << std::endl;
        
        if (use_initial_guess_)
        {
            std::cout << "\n--- Initial Guess Configuration ---" << std::endl;
            std::cout << "Initial camera position (Base frame):" << std::endl;
            std::cout << "  X: " << initial_guess_position_[0] << " m" << std::endl;
            std::cout << "  Y: " << initial_guess_position_[1] << " m" << std::endl;
            std::cout << "  Z: " << initial_guess_position_[2] << " m" << std::endl;
            std::cout << "Outlier threshold: " << outlier_threshold_ * 1000 << " mm" << std::endl;
        }
        
        std::cout << "========================================\n" << std::endl;
    }

    bool loadCameraCalibration(const std::string& filename)
    {
        cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "Failed to open calibration file: " << filename << std::endl;
            return false;
        }

        fs["camera_matrix"] >> camera_matrix_;
        fs["distortion_coefficients"] >> dist_coeffs_;
        fs.release();

        if (camera_matrix_.empty() || dist_coeffs_.empty())
        {
            std::cerr << "Invalid calibration data in file!" << std::endl;
            return false;
        }

        camera_calibrated_ = true;
        return true;
    }

    std::string trim(const std::string& str)
    {
        size_t first = str.find_first_not_of(" \t\r\n");
        if (first == std::string::npos)
            return "";
        size_t last = str.find_last_not_of(" \t\r\n");
        return str.substr(first, (last - first + 1));
    }

    std::string extractValue(const std::string& line, const std::string& prefix)
    {
        size_t pos = line.find(prefix);
        if (pos != std::string::npos)
        {
            return trim(line.substr(pos + prefix.length()));
        }
        return "";
    }

    std::vector<double> parseCommaSeparated(const std::string& str)
    {
        std::vector<double> values;
        std::stringstream ss(str);
        std::string token;
        
        while (std::getline(ss, token, ','))
        {
            try
            {
                values.push_back(std::stod(trim(token)));
            }
            catch (...)
            {
                std::cerr << "Failed to parse value: " << token << std::endl;
            }
        }
        
        return values;
    }

    bool loadTCPPoses()
    {
        std::ifstream file(tcp_poses_file_);
        if (!file.is_open())
        {
            std::cerr << "ERROR: Cannot open TCP poses file: " << tcp_poses_file_ << std::endl;
            return false;
        }

        std::cout << "\nLoading TCP poses from: " << tcp_poses_file_ << std::endl;
        std::cout << "(These represent: Robot Base → ChArUco Board transformations)" << std::endl;
        
        std::string line;
        int pose_count = 0;
        RobotPose current_pose;
        int line_in_block = 0;
        
        while (std::getline(file, line))
        {
            line = trim(line);
            
            if (line.empty())
                continue;
            
            if (line.find("====") != std::string::npos)
            {
                if (!current_pose.image_name.empty())
                {
                    robot_poses_.push_back(current_pose);
                    
                    fs::path img_path(current_pose.image_name);
                    std::string filename = img_path.filename().string();
                    robot_poses_map_[filename] = current_pose;
                    
                    pose_count++;
                    
                    std::cout << "  Loaded pose " << pose_count << ": " << filename 
                              << " | Pos: [" << current_pose.position[0] << ", "
                              << current_pose.position[1] << ", "
                              << current_pose.position[2] << "]" << std::endl;
                }
                
                current_pose = RobotPose();
                line_in_block = 0;
                continue;
            }
            
            if (line.find("Zeitstempel:") != std::string::npos)
            {
                current_pose.timestamp = extractValue(line, "Zeitstempel:");
            }
            else if (line.find("Position:") != std::string::npos)
            {
                std::string pos_str = extractValue(line, "Position:");
                std::vector<double> values = parseCommaSeparated(pos_str);
                
                if (values.size() == 3)
                {
                    current_pose.position = cv::Vec3d(values[0], values[1], values[2]);
                }
                else
                {
                    std::cerr << "WARNING: Invalid position format: " << pos_str << std::endl;
                }
            }
            else if (line.find("Orientierung") != std::string::npos || 
                     line.find("Quaternion") != std::string::npos)
            {
                std::string quat_str = extractValue(line, ":");
                std::vector<double> values = parseCommaSeparated(quat_str);
                
                if (values.size() == 4)
                {
                    current_pose.quaternion = cv::Vec4d(values[0], values[1], values[2], values[3]);
                }
                else
                {
                    std::cerr << "WARNING: Invalid quaternion format: " << quat_str << std::endl;
                }
            }
            else if (line.find("Bild:") != std::string::npos)
            {
                current_pose.image_name = extractValue(line, "Bild:");
            }
        }
        
        if (!current_pose.image_name.empty())
        {
            robot_poses_.push_back(current_pose);
            
            fs::path img_path(current_pose.image_name);
            std::string filename = img_path.filename().string();
            robot_poses_map_[filename] = current_pose;
            
            pose_count++;
            
            std::cout << "  Loaded pose " << pose_count << ": " << filename 
                      << " | Pos: [" << current_pose.position[0] << ", "
                      << current_pose.position[1] << ", "
                      << current_pose.position[2] << "]" << std::endl;
        }
        
        file.close();
        
        std::cout << "Total TCP poses loaded: " << robot_poses_.size() << std::endl;
        return !robot_poses_.empty();
    }

    void processAllImages()
    {
        if (!fs::exists(input_dir_))
        {
            std::cerr << "ERROR: Directory does not exist: " << input_dir_ << std::endl;
            return;
        }

        std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
        int processed_count = 0;
        int detected_count = 0;

        std::cout << "\nProcessing images in: " << input_dir_ << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        // Collect and sort image files
        std::vector<std::string> image_files;
        for (const auto& entry : fs::directory_iterator(input_dir_))
        {
            if (!entry.is_regular_file())
                continue;

            std::string filepath = entry.path().string();
            std::string extension = entry.path().extension().string();
            
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end() &&
                filepath.find("_processed") == std::string::npos &&
                filepath.find("_origin") == std::string::npos)
            {
                image_files.push_back(filepath);
            }
        }
        
        std::sort(image_files.begin(), image_files.end());
        
        // Process images
        for (size_t i = 0; i < image_files.size(); i++)
        {
            const std::string& filepath = image_files[i];
            std::cout << "\nProcessing [" << (i+1) << "/" << image_files.size() << "]: " 
                      << fs::path(filepath).filename() << std::endl;
            
            if (processImage(filepath, i))
            {
                detected_count++;
            }
            processed_count++;
        }

        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing complete!" << std::endl;
        std::cout << "Total images processed: " << processed_count << std::endl;
        std::cout << "Boards detected: " << detected_count << std::endl;
        std::cout << "Origin images created: " << detected_count << std::endl;
        
        // Perform eye-to-hand calibration if we have enough data
        if (detected_count >= 3 && !robot_poses_.empty())
        {
            performEyeToHandCalibration();
        }
        else if (!robot_poses_.empty())
        {
            std::cout << "\nWARNING: Not enough valid detections for calibration." << std::endl;
            std::cout << "Need at least 3, got " << detected_count << std::endl;
        }
        
        std::cout << "========================================" << std::endl;
    }

    bool processImage(const std::string& filepath, size_t image_index)
    {
        cv::Mat image = cv::imread(filepath);
        if (image.empty())
        {
            std::cerr << "  ERROR: Failed to load image!" << std::endl;
            return false;
        }

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        
        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, parameters_, rejected);

        cv::Mat processed_image = image.clone();
        bool board_detected = false;

        if (!ids.empty())
        {
            std::cout << "  Found " << ids.size() << " ArUco markers" << std::endl;
            cv::aruco::drawDetectedMarkers(processed_image, corners, ids);

            if (ids.size() > 4)
            {
                std::vector<cv::Point2f> charuco_corners;
                std::vector<int> charuco_ids;
                
                int interpolated = cv::aruco::interpolateCornersCharuco(
                    corners, ids, gray, board_, charuco_corners, charuco_ids
                );

                std::cout << "  Interpolated " << interpolated << " ChArUco corners" << std::endl;

                if (interpolated >= MIN_CORNERS_THRESHOLD)
                {
                    cv::aruco::drawDetectedCornersCharuco(processed_image, charuco_corners, charuco_ids);

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
                        cv::drawFrameAxes(processed_image, camera_matrix_, dist_coeffs_, 
                                         rvec, tvec, 0.05f, 3);

                        std::string pos_text = cv::format("Camera→Board: [%.3f, %.3f, %.3f] m", 
                                                         tvec[0], tvec[1], tvec[2]);
                        cv::putText(processed_image, pos_text, 
                                   cv::Point(10, 30), 
                                   cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   cv::Scalar(0, 255, 0), 
                                   2);

                        std::string corner_text = cv::format("Corners: %d", interpolated);
                        cv::putText(processed_image, corner_text, 
                                   cv::Point(10, 60), 
                                   cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   cv::Scalar(0, 255, 0), 
                                   2);

                        std::cout << "  Board detected successfully!" << std::endl;
                        std::cout << "  Camera→Board: [" << tvec[0] << ", " << tvec[1] << ", " << tvec[2] << "]" << std::endl;
                        
                        board_detected = true;

                        // Store detected pose for calibration
                        DetectedPose det_pose;
                        det_pose.image_name = fs::path(filepath).filename().string();
                        det_pose.rvec = rvec;
                        det_pose.tvec = tvec;
                        detected_poses_.push_back(det_pose);

                        // Create origin image
                        cv::Mat origin_image = image.clone();
                        cv::drawFrameAxes(origin_image, camera_matrix_, dist_coeffs_, 
                                         rvec, tvec, 0.05f, 3);

                        cv::putText(origin_image, "ChArUco Board Origin", 
                                   cv::Point(10, 30), 
                                   cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   cv::Scalar(0, 255, 0), 
                                   2);

                        std::string origin_path = generateOutputFilename(filepath, "_origin");
                        if (cv::imwrite(origin_path, origin_image))
                        {
                            std::cout << "  Saved (origin): " << fs::path(origin_path).filename() << std::endl;
                        }
                        else
                        {
                            std::cerr << "  ERROR: Failed to save origin image!" << std::endl;
                        }
                    }
                    else
                    {
                        std::cout << "  WARNING: Pose estimation failed - no origin image created" << std::endl;
                        cv::putText(processed_image, "Pose estimation failed", 
                                   cv::Point(10, 30), 
                                   cv::FONT_HERSHEY_SIMPLEX, 
                                   0.6, 
                                   cv::Scalar(0, 165, 255), 
                                   2);
                    }
                }
                else
                {
                    std::cout << "  WARNING: Too few corners (" << interpolated << " < " 
                              << MIN_CORNERS_THRESHOLD << ") - no origin image created" << std::endl;
                    
                    std::string warning = cv::format("Too few corners: %d < %d", 
                                                     interpolated, MIN_CORNERS_THRESHOLD);
                    cv::putText(processed_image, warning, 
                               cv::Point(10, 30), 
                               cv::FONT_HERSHEY_SIMPLEX, 
                               0.6, 
                               cv::Scalar(0, 0, 255), 
                               2);
                }
            }
            else
            {
                std::cout << "  WARNING: Too few markers for ChArUco interpolation" << std::endl;
                cv::putText(processed_image, "Too few markers for board detection", 
                           cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 
                           0.6, 
                           cv::Scalar(0, 0, 255), 
                           2);
            }
        }
        else
        {
            std::cout << "  No ArUco markers detected" << std::endl;
            cv::putText(processed_image, "No markers detected", 
                       cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 
                       0.6, 
                       cv::Scalar(0, 0, 255), 
                       2);
        }

        std::string processed_path = generateOutputFilename(filepath, "_processed");
        if (cv::imwrite(processed_path, processed_image))
        {
            std::cout << "  Saved (processed): " << fs::path(processed_path).filename() << std::endl;
        }
        else
        {
            std::cerr << "  ERROR: Failed to save processed image!" << std::endl;
        }

        return board_detected;
    }

    void performEyeToHandCalibration()
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "EYE-TO-HAND CALIBRATION" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Setup: Camera FIXED in environment" << std::endl;
        std::cout << "       ChArUco Board mounted on Robot TCP" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Compute all transformations
        std::vector<MeasurementQuality> measurements;
        
        for (size_t i = 0; i < detected_poses_.size(); i++)
        {
            const DetectedPose& det_pose = detected_poses_[i];
            
            if (robot_poses_map_.find(det_pose.image_name) != robot_poses_map_.end())
            {
                const RobotPose& robot_pose = robot_poses_map_[det_pose.image_name];
                
                // T_base2board: Robot Base → ChArUco Board
                cv::Mat T_base2board = robot_pose.getTransformationMatrix();
                
                // T_camera2board: Camera → ChArUco Board
                cv::Mat T_camera2board = det_pose.getTransformationMatrix();
                
                // Calculate: T_base2camera = T_base2board * inv(T_camera2board)
                cv::Mat T_camera2board_inv = T_camera2board.inv();
                cv::Mat T_base2camera = T_base2board * T_camera2board_inv;
                
                // Calculate deviation from initial guess (if provided)
                double deviation = 0.0;
                if (use_initial_guess_)
                {
                    cv::Vec3d measured_pos(
                        T_base2camera.at<double>(0, 3),
                        T_base2camera.at<double>(1, 3),
                        T_base2camera.at<double>(2, 3)
                    );
                    
                    cv::Vec3d diff = measured_pos - initial_guess_position_;
                    deviation = cv::norm(diff);
                }
                
                MeasurementQuality mq;
                mq.image_name = det_pose.image_name;
                mq.T_base2camera = T_base2camera;
                mq.deviation_from_guess = deviation;
                mq.is_outlier = false;
                mq.weight = 1.0;
                
                measurements.push_back(mq);
                
                std::cout << "\nImage " << (i+1) << ": " << det_pose.image_name << std::endl;
                std::cout << "  Base→Board: [" 
                          << robot_pose.position[0] << ", "
                          << robot_pose.position[1] << ", "
                          << robot_pose.position[2] << "]" << std::endl;
                std::cout << "  Camera→Board: ["
                          << det_pose.tvec[0] << ", "
                          << det_pose.tvec[1] << ", "
                          << det_pose.tvec[2] << "]" << std::endl;
                std::cout << "  Computed Base→Camera: [" 
                          << T_base2camera.at<double>(0,3) << ", "
                          << T_base2camera.at<double>(1,3) << ", "
                          << T_base2camera.at<double>(2,3) << "]" << std::endl;
                
                if (use_initial_guess_)
                {
                    std::cout << "  Deviation from guess: " << deviation * 1000 << " mm" << std::endl;
                }
            }
            else
            {
                std::cerr << "\nWARNING: No robot pose found for " << det_pose.image_name << std::endl;
            }
        }
        
        if (measurements.empty())
        {
            std::cerr << "\n========================================" << std::endl;
            std::cerr << "ERROR: No valid pose pairs!" << std::endl;
            std::cerr << "========================================" << std::endl;
            return;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Total measurements: " << measurements.size() << std::endl;
        
        // Outlier detection if initial guess provided
        int outliers_removed = 0;
        if (use_initial_guess_ && measurements.size() >= 5)
        {
            std::cout << "\n--- Outlier Detection ---" << std::endl;
            std::cout << "Threshold: " << outlier_threshold_ * 1000 << " mm" << std::endl;
            
            for (auto& mq : measurements)
            {
                if (mq.deviation_from_guess > outlier_threshold_)
                {
                    mq.is_outlier = true;
                    mq.weight = 0.0;
                    outliers_removed++;
                    std::cout << "  OUTLIER: " << mq.image_name 
                              << " (deviation: " << mq.deviation_from_guess * 1000 << " mm)" << std::endl;
                }
            }
            
            std::cout << "Outliers removed: " << outliers_removed << std::endl;
            std::cout << "Valid measurements: " << (measurements.size() - outliers_removed) << std::endl;
        }
        
        // Calculate weighted average (or use robust estimation)
        std::vector<cv::Mat> valid_transforms;
        std::vector<std::string> valid_images;
        std::vector<double> weights;
        
        for (const auto& mq : measurements)
        {
            if (!mq.is_outlier)
            {
                valid_transforms.push_back(mq.T_base2camera);
                valid_images.push_back(mq.image_name);
                
                // Inverse distance weighting if initial guess provided
                double weight = 1.0;
                if (use_initial_guess_ && mq.deviation_from_guess > 1e-6)
                {
                    // Gaussian weighting: weight decreases with distance from guess
                    double sigma = outlier_threshold_ / 3.0;  // 3-sigma rule
                    weight = std::exp(-0.5 * std::pow(mq.deviation_from_guess / sigma, 2));
                }
                weights.push_back(weight);
            }
        }
        
        if (valid_transforms.empty())
        {
            std::cerr << "\nERROR: All measurements were outliers!" << std::endl;
            return;
        }
        
        if (valid_transforms.size() < 3)
        {
            std::cerr << "\nWARNING: Less than 3 valid measurements!" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        
        // Consistency check
        analyzeConsistency(valid_transforms);
        
        // Compute weighted average
        cv::Mat T_base2camera_avg;
        if (use_initial_guess_)
        {
            T_base2camera_avg = weightedAverageTransformations(valid_transforms, weights);
            std::cout << "\nUsing WEIGHTED averaging based on initial guess" << std::endl;
        }
        else
        {
            T_base2camera_avg = averageTransformations(valid_transforms);
            std::cout << "\nUsing SIMPLE averaging (no initial guess)" << std::endl;
        }
        
        // Extract rotation and translation
        cv::Mat R_base2camera = T_base2camera_avg(cv::Rect(0, 0, 3, 3)).clone();
        cv::Mat t_base2camera = T_base2camera_avg(cv::Rect(3, 0, 1, 3)).clone();
        
        // Check against initial guess
        if (use_initial_guess_)
        {
            cv::Vec3d final_pos(
                t_base2camera.at<double>(0),
                t_base2camera.at<double>(1),
                t_base2camera.at<double>(2)
            );
            cv::Vec3d diff = final_pos - initial_guess_position_;
            double final_deviation = cv::norm(diff);
            
            std::cout << "\n--- Validation Against Initial Guess ---" << std::endl;
            std::cout << "Initial guess: [" 
                      << initial_guess_position_[0] << ", "
                      << initial_guess_position_[1] << ", "
                      << initial_guess_position_[2] << "]" << std::endl;
            std::cout << "Final result:  [" 
                      << final_pos[0] << ", "
                      << final_pos[1] << ", "
                      << final_pos[2] << "]" << std::endl;
            std::cout << "Difference:    [" 
                      << diff[0] << ", "
                      << diff[1] << ", "
                      << diff[2] << "]" << std::endl;
            std::cout << "Total deviation: " << final_deviation * 1000 << " mm" << std::endl;
            
            if (final_deviation > outlier_threshold_)
            {
                std::cout << "\n⚠️  WARNING: Final result deviates significantly from initial guess!" << std::endl;
                std::cout << "   Please verify:" << std::endl;
                std::cout << "   - Initial guess accuracy" << std::endl;
                std::cout << "   - Camera calibration" << std::endl;
                std::cout << "   - Robot pose accuracy" << std::endl;
                std::cout << "   - ChArUco detection quality" << std::endl;
            }
            else
            {
                std::cout << "\n✓ Result is consistent with initial guess" << std::endl;
            }
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "*** FINAL RESULT ***" << std::endl;
        std::cout << "Robot Base → Camera Transformation" << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::cout << "\nTransformation Matrix (4x4):" << std::endl;
        std::cout << T_base2camera_avg << std::endl;
        
        std::cout << "\nRotation Matrix (3x3):" << std::endl;
        std::cout << R_base2camera << std::endl;
        
        std::cout << "\nTranslation Vector [x, y, z] (meters):" << std::endl;
        std::cout << "[" << t_base2camera.at<double>(0) << ", " 
                  << t_base2camera.at<double>(1) << ", " 
                  << t_base2camera.at<double>(2) << "]" << std::endl;
        
        // Convert to quaternion and RPY
        cv::Vec3d rvec;
        cv::Rodrigues(R_base2camera, rvec);
        
        std::cout << "\nRotation Vector (Rodrigues) [rx, ry, rz]:" << std::endl;
        std::cout << "[" << rvec[0] << ", " << rvec[1] << ", " << rvec[2] << "]" << std::endl;
        
        double angle = cv::norm(rvec);
        if (angle > 1e-6)
        {
            cv::Vec3d axis = rvec / angle;
            double qw = std::cos(angle / 2.0);
            double qx = axis[0] * std::sin(angle / 2.0);
            double qy = axis[1] * std::sin(angle / 2.0);
            double qz = axis[2] * std::sin(angle / 2.0);
            
            std::cout << "\nQuaternion [x, y, z, w]:" << std::endl;
            std::cout << "[" << qx << ", " << qy << ", " << qz << ", " << qw << "]" << std::endl;
            
            // Roll-Pitch-Yaw
            double roll = std::atan2(R_base2camera.at<double>(2,1), R_base2camera.at<double>(2,2));
            double pitch = std::atan2(-R_base2camera.at<double>(2,0), 
                                      std::sqrt(R_base2camera.at<double>(2,1)*R_base2camera.at<double>(2,1) + 
                                               R_base2camera.at<double>(2,2)*R_base2camera.at<double>(2,2)));
            double yaw = std::atan2(R_base2camera.at<double>(1,0), R_base2camera.at<double>(0,0));
            
            std::cout << "\nRoll-Pitch-Yaw [rad]:" << std::endl;
            std::cout << "[" << roll << ", " << pitch << ", " << yaw << "]" << std::endl;
            
            std::cout << "\nRoll-Pitch-Yaw [deg]:" << std::endl;
            std::cout << "[" << roll*180/M_PI << ", " << pitch*180/M_PI << ", " << yaw*180/M_PI << "]" << std::endl;
            
            // For ROS/URDF
            std::cout << "\n--- For ROS2/URDF Static Transform ---" << std::endl;
            std::cout << "<origin xyz=\"" 
                      << t_base2camera.at<double>(0) << " " 
                      << t_base2camera.at<double>(1) << " " 
                      << t_base2camera.at<double>(2) << "\"" << std::endl;
            std::cout << "        rpy=\"" << roll << " " << pitch << " " << yaw << "\"/>" << std::endl;
            
            std::cout << "\n--- ROS2 static_transform_publisher command ---" << std::endl;
            std::cout << "ros2 run tf2_ros static_transform_publisher \\" << std::endl;
            std::cout << "  " << t_base2camera.at<double>(0) << " "
                      << t_base2camera.at<double>(1) << " "
                      << t_base2camera.at<double>(2) << " \\" << std::endl;
            std::cout << "  " << qx << " " << qy << " " << qz << " " << qw << " \\" << std::endl;
            std::cout << "  base_link camera_link" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        
        // Save results
        saveCalibrationResult("eye_to_hand", R_base2camera, t_base2camera, T_base2camera_avg, 
                             valid_images, measurements, use_initial_guess_, initial_guess_position_);
        
        std::cout << "\n✓ Calibration complete!" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    void analyzeConsistency(const std::vector<cv::Mat>& transforms)
    {
        if (transforms.size() < 2)
            return;
        
        std::cout << "\n--- Consistency Analysis ---" << std::endl;
        
        // Calculate statistics
        cv::Mat mean_t = cv::Mat::zeros(3, 1, CV_64F);
        std::vector<cv::Vec3d> positions;
        
        for (const auto& T : transforms)
        {
            cv::Vec3d pos(T.at<double>(0,3), T.at<double>(1,3), T.at<double>(2,3));
            positions.push_back(pos);
            mean_t.at<double>(0) += pos[0];
            mean_t.at<double>(1) += pos[1];
            mean_t.at<double>(2) += pos[2];
        }
        
        mean_t /= static_cast<double>(transforms.size());
        
        // Standard deviation
        double var_x = 0, var_y = 0, var_z = 0;
        double max_dev = 0;
        
        for (const auto& pos : positions)
        {
            double dx = pos[0] - mean_t.at<double>(0);
            double dy = pos[1] - mean_t.at<double>(1);
            double dz = pos[2] - mean_t.at<double>(2);
            
            var_x += dx * dx;
            var_y += dy * dy;
            var_z += dz * dz;
            
            double dev = std::sqrt(dx*dx + dy*dy + dz*dz);
            max_dev = std::max(max_dev, dev);
        }
        
        var_x /= transforms.size();
        var_y /= transforms.size();
        var_z /= transforms.size();
        
        double std_x = std::sqrt(var_x);
        double std_y = std::sqrt(var_y);
        double std_z = std::sqrt(var_z);
        double std_total = std::sqrt(var_x + var_y + var_z);
        
        std::cout << "Standard Deviation:" << std::endl;
        std::cout << "  X: " << std_x * 1000 << " mm" << std::endl;
        std::cout << "  Y: " << std_y * 1000 << " mm" << std::endl;
        std::cout << "  Z: " << std_z * 1000 << " mm" << std::endl;
        std::cout << "  Total: " << std_total * 1000 << " mm" << std::endl;
        std::cout << "Max deviation from mean: " << max_dev * 1000 << " mm" << std::endl;
        
        if (std_total > 0.05)  // 50mm
        {
            std::cout << "\n⚠️  WARNING: High variance detected!" << std::endl;
            std::cout << "   Possible causes:" << std::endl;
            std::cout << "   - Poor camera calibration" << std::endl;
            std::cout << "   - Inaccurate robot poses" << std::endl;
            std::cout << "   - Low-quality ChArUco detections" << std::endl;
            std::cout << "   - Moving camera or board mounting issues" << std::endl;
        }
        else if (std_total > 0.01)  // 10mm
        {
            std::cout << "\n✓ Moderate consistency (acceptable for most applications)" << std::endl;
        }
        else
        {
            std::cout << "\n✓ Excellent consistency!" << std::endl;
        }
    }

    cv::Mat weightedAverageTransformations(const std::vector<cv::Mat>& transforms, 
                                           const std::vector<double>& weights)
    {
        if (transforms.empty())
        {
            return cv::Mat::eye(4, 4, CV_64F);
        }
        
        if (transforms.size() != weights.size())
        {
            std::cerr << "ERROR: transforms and weights size mismatch!" << std::endl;
            return averageTransformations(transforms);
        }
        
        // Normalize weights
        double weight_sum = 0.0;
        for (double w : weights)
            weight_sum += w;
        
        if (weight_sum < 1e-10)
        {
            return averageTransformations(transforms);
        }
        
        // Weighted average translation
        cv::Mat avg_translation = cv::Mat::zeros(3, 1, CV_64F);
        std::vector<cv::Mat> rotations;
        std::vector<double> norm_weights;
        
        for (size_t i = 0; i < transforms.size(); i++)
        {
            double w = weights[i] / weight_sum;
            norm_weights.push_back(w);
            
            avg_translation += transforms[i](cv::Rect(3, 0, 1, 3)) * w;
            rotations.push_back(transforms[i](cv::Rect(0, 0, 3, 3)).clone());
        }
        
        // Weighted average rotations
        cv::Mat avg_rotation = weightedAverageRotations(rotations, norm_weights);
        
        // Combine
        cv::Mat result = cv::Mat::eye(4, 4, CV_64F);
        avg_rotation.copyTo(result(cv::Rect(0, 0, 3, 3)));
        avg_translation.copyTo(result(cv::Rect(3, 0, 1, 3)));
        
        return result;
    }

    cv::Mat weightedAverageRotations(const std::vector<cv::Mat>& rotations, 
                                     const std::vector<double>& weights)
    {
        if (rotations.empty())
        {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        if (rotations.size() == 1)
        {
            return rotations[0].clone();
        }
        
        // Convert to quaternions
        std::vector<cv::Vec4d> quaternions;
        
        for (const auto& R : rotations)
        {
            cv::Vec3d rvec;
            cv::Rodrigues(R, rvec);
            double angle = cv::norm(rvec);
            
            if (angle > 1e-6)
            {
                cv::Vec3d axis = rvec / angle;
                double qw = std::cos(angle / 2.0);
                double qx = axis[0] * std::sin(angle / 2.0);
                double qy = axis[1] * std::sin(angle / 2.0);
                double qz = axis[2] * std::sin(angle / 2.0);
                
                quaternions.push_back(cv::Vec4d(qx, qy, qz, qw));
            }
            else
            {
                quaternions.push_back(cv::Vec4d(0, 0, 0, 1));
            }
        }
        
        // Ensure same hemisphere
        for (size_t i = 1; i < quaternions.size(); i++)
        {
            if (quaternions[0].dot(quaternions[i]) < 0)
            {
                quaternions[i] = -quaternions[i];
            }
        }
        
        // Weighted average
        cv::Vec4d avg_q(0, 0, 0, 0);
        for (size_t i = 0; i < quaternions.size(); i++)
        {
            avg_q += quaternions[i] * weights[i];
        }
        
        // Normalize
        double norm = std::sqrt(avg_q[0]*avg_q[0] + avg_q[1]*avg_q[1] + 
                               avg_q[2]*avg_q[2] + avg_q[3]*avg_q[3]);
        
        if (norm < 1e-6)
        {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        avg_q /= norm;
        
        // Convert to rotation matrix
        double qx = avg_q[0], qy = avg_q[1], qz = avg_q[2], qw = avg_q[3];
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        
        R.at<double>(0, 0) = 1 - 2*qy*qy - 2*qz*qz;
        R.at<double>(0, 1) = 2*qx*qy - 2*qz*qw;
        R.at<double>(0, 2) = 2*qx*qz + 2*qy*qw;
        
        R.at<double>(1, 0) = 2*qx*qy + 2*qz*qw;
        R.at<double>(1, 1) = 1 - 2*qx*qx - 2*qz*qz;
        R.at<double>(1, 2) = 2*qy*qz - 2*qx*qw;
        
        R.at<double>(2, 0) = 2*qx*qz - 2*qy*qw;
        R.at<double>(2, 1) = 2*qy*qz + 2*qx*qw;
        R.at<double>(2, 2) = 1 - 2*qx*qx - 2*qy*qy;
        
        return R;
    }

    cv::Mat averageTransformations(const std::vector<cv::Mat>& transforms)
    {
        if (transforms.empty())
        {
            return cv::Mat::eye(4, 4, CV_64F);
        }
        
        // Average translation
        cv::Mat avg_translation = cv::Mat::zeros(3, 1, CV_64F);
        std::vector<cv::Mat> rotations;
        
        for (const auto& T : transforms)
        {
            avg_translation += T(cv::Rect(3, 0, 1, 3));
            rotations.push_back(T(cv::Rect(0, 0, 3, 3)).clone());
        }
        
        avg_translation /= static_cast<double>(transforms.size());
        
        // Average rotations
        cv::Mat avg_rotation = averageRotations(rotations);
        
        // Combine
        cv::Mat result = cv::Mat::eye(4, 4, CV_64F);
        avg_rotation.copyTo(result(cv::Rect(0, 0, 3, 3)));
        avg_translation.copyTo(result(cv::Rect(3, 0, 1, 3)));
        
        return result;
    }

    cv::Mat averageRotations(const std::vector<cv::Mat>& rotations)
    {
        if (rotations.empty())
        {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        if (rotations.size() == 1)
        {
            return rotations[0].clone();
        }
        
        // Convert all rotations to quaternions
        std::vector<cv::Vec4d> quaternions;
        
        for (const auto& R : rotations)
        {
            cv::Vec3d rvec;
            cv::Rodrigues(R, rvec);
            double angle = cv::norm(rvec);
            
            if (angle > 1e-6)
            {
                cv::Vec3d axis = rvec / angle;
                double qw = std::cos(angle / 2.0);
                double qx = axis[0] * std::sin(angle / 2.0);
                double qy = axis[1] * std::sin(angle / 2.0);
                double qz = axis[2] * std::sin(angle / 2.0);
                
                quaternions.push_back(cv::Vec4d(qx, qy, qz, qw));
            }
            else
            {
                quaternions.push_back(cv::Vec4d(0, 0, 0, 1));
            }
        }
        
        // Ensure all in same hemisphere
        for (size_t i = 1; i < quaternions.size(); i++)
        {
            if (quaternions[0].dot(quaternions[i]) < 0)
            {
                quaternions[i] = -quaternions[i];
            }
        }
        
        // Average
        cv::Vec4d avg_q(0, 0, 0, 0);
        for (const auto& q : quaternions)
        {
            avg_q += q;
        }
        avg_q /= static_cast<double>(quaternions.size());
        
        // Normalize
        double norm = std::sqrt(avg_q[0]*avg_q[0] + avg_q[1]*avg_q[1] + 
                               avg_q[2]*avg_q[2] + avg_q[3]*avg_q[3]);
        
        if (norm < 1e-6)
        {
            return cv::Mat::eye(3, 3, CV_64F);
        }
        
        avg_q /= norm;
        
        // Convert back to rotation matrix
        double qx = avg_q[0], qy = avg_q[1], qz = avg_q[2], qw = avg_q[3];
        cv::Mat R = cv::Mat::eye(3, 3, CV_64F);
        
        R.at<double>(0, 0) = 1 - 2*qy*qy - 2*qz*qz;
        R.at<double>(0, 1) = 2*qx*qy - 2*qz*qw;
        R.at<double>(0, 2) = 2*qx*qz + 2*qy*qw;
        
        R.at<double>(1, 0) = 2*qx*qy + 2*qz*qw;
        R.at<double>(1, 1) = 1 - 2*qx*qx - 2*qz*qz;
        R.at<double>(1, 2) = 2*qy*qz - 2*qx*qw;
        
        R.at<double>(2, 0) = 2*qx*qz - 2*qy*qw;
        R.at<double>(2, 1) = 2*qy*qz + 2*qx*qw;
        R.at<double>(2, 2) = 1 - 2*qx*qx - 2*qy*qy;
        
        return R;
    }

    void saveCalibrationResult(const std::string& method_name, 
                                const cv::Mat& R, 
                                const cv::Mat& t,
                                const cv::Mat& T,
                                const std::vector<std::string>& images,
                                const std::vector<MeasurementQuality>& measurements,
                                bool used_initial_guess,
                                const cv::Vec3d& initial_guess)
    {
        std::string filename = input_dir_ + "/calibration_result_" + method_name + ".yaml";
        
        cv::FileStorage fs(filename, cv::FileStorage::WRITE);
        if (!fs.isOpened())
        {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return;
        }
        
        fs << "calibration_type" << "eye_to_hand";
        fs << "description" << "Camera fixed in environment, ChArUco board on robot TCP";
        fs << "result" << "Robot Base to Camera transformation";
        fs << "num_images_used" << static_cast<int>(images.size());
        fs << "num_images_total" << static_cast<int>(measurements.size());
        
        if (used_initial_guess)
        {
            fs << "used_initial_guess" << 1;
            fs << "initial_guess_x" << initial_guess[0];
            fs << "initial_guess_y" << initial_guess[1];
            fs << "initial_guess_z" << initial_guess[2];
            
            // Calculate final deviation
            cv::Vec3d final_pos(t.at<double>(0), t.at<double>(1), t.at<double>(2));
            cv::Vec3d diff = final_pos - initial_guess;
            double deviation = cv::norm(diff);
            
            fs << "deviation_from_guess_m" << deviation;
            fs << "deviation_from_guess_mm" << (deviation * 1000);
        }
        else
        {
            fs << "used_initial_guess" << 0;
        }
        
        // Save image list with quality metrics
        fs << "measurements" << "[";
        for (const auto& mq : measurements)
        {
            fs << "{";
            fs << "image" << mq.image_name;
            fs << "is_outlier" << (mq.is_outlier ? 1 : 0);
            fs << "weight" << mq.weight;
            if (used_initial_guess)
            {
                fs << "deviation_from_guess_m" << mq.deviation_from_guess;
                fs << "deviation_from_guess_mm" << (mq.deviation_from_guess * 1000);
            }
            fs << "}";
        }
        fs << "]";
        
        // Save transformation
        fs << "base_to_camera_rotation" << R;
        fs << "base_to_camera_translation" << t;
        fs << "base_to_camera_transform" << T;
        
        // Save quaternion
        cv::Vec3d rvec;
        cv::Rodrigues(R, rvec);
        double angle = cv::norm(rvec);
        if (angle > 1e-6)
        {
            cv::Vec3d axis = rvec / angle;
            double qw = std::cos(angle / 2.0);
            double qx = axis[0] * std::sin(angle / 2.0);
            double qy = axis[1] * std::sin(angle / 2.0);
            double qz = axis[2] * std::sin(angle / 2.0);
            
            fs << "quaternion_x" << qx;
            fs << "quaternion_y" << qy;
            fs << "quaternion_z" << qz;
            fs << "quaternion_w" << qw;
            
            // Roll-Pitch-Yaw
            double roll = std::atan2(R.at<double>(2,1), R.at<double>(2,2));
            double pitch = std::atan2(-R.at<double>(2,0), 
                                      std::sqrt(R.at<double>(2,1)*R.at<double>(2,1) + 
                                               R.at<double>(2,2)*R.at<double>(2,2)));
            double yaw = std::atan2(R.at<double>(1,0), R.at<double>(0,0));
            
            fs << "roll_rad" << roll;
            fs << "pitch_rad" << pitch;
            fs << "yaw_rad" << yaw;
            fs << "roll_deg" << (roll * 180.0 / M_PI);
            fs << "pitch_deg" << (pitch * 180.0 / M_PI);
            fs << "yaw_deg" << (yaw * 180.0 / M_PI);
        }
        
        fs.release();
        
        std::cout << "\n✓ Results saved to: " << filename << std::endl;
    }

private:
    std::string generateOutputFilename(const std::string& input_path, const std::string& suffix)
    {
        fs::path p(input_path);
        std::string stem = p.stem().string();
        std::string extension = p.extension().string();
        std::string parent = p.parent_path().string();
        
        return parent + "/" + stem + suffix + extension;
    }

    int SQUARES_X;
    int SQUARES_Y;
    float SQUARE_LENGTH;
    float MARKER_LENGTH;
    int MIN_CORNERS_THRESHOLD;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::CharucoBoard> board_;
    cv::Ptr<cv::aruco::DetectorParameters> parameters_;

    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    bool camera_calibrated_;

    std::string input_dir_;
    std::string tcp_poses_file_;
    
    std::vector<RobotPose> robot_poses_;
    std::map<std::string, RobotPose> robot_poses_map_;
    std::vector<DetectedPose> detected_poses_;
    
    // Initial guess parameters
    cv::Vec3d initial_guess_position_;
    bool use_initial_guess_;
    double outlier_threshold_;
};

int main(int argc, char** argv)
{
    std::string input_dir = "/";
    std::string tcp_poses_file = "/ros2_ws/src/inspire_hand_Linkage/src/cam_calibrate/tcp_poses.txt";
    std::string calib_file = "";
    
    // Initial guess parameters
    bool use_initial_guess = false;
    cv::Vec3d initial_guess(0, 0, 0);
    double outlier_threshold = 0.1;  // 100mm default

    // Parse command line arguments
    if (argc > 1)
        input_dir = argv[1];
    if (argc > 2)
        tcp_poses_file = argv[2];
    if (argc > 3)
        calib_file = argv[3];
    
    // Check for initial guess (optional arguments)
    if (argc > 6)
    {
        use_initial_guess = true;
        initial_guess[0] = std::stod(argv[4]);  // x
        initial_guess[1] = std::stod(argv[5]);  // y
        initial_guess[2] = std::stod(argv[6]);  // z
        
        if (argc > 7)
        {
            outlier_threshold = std::stod(argv[7]);
        }
    }

    std::cout << "========================================" << std::endl;
    std::cout << "ChArUco Eye-to-Hand Calibration" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Camera: FIXED in environment" << std::endl;
    std::cout << "Board:  MOVING (mounted on robot TCP)" << std::endl;
    std::cout << "Result: Robot Base → Camera transformation" << std::endl;
    
    if (use_initial_guess)
    {
        std::cout << "\n✓ Using initial guess for outlier detection" << std::endl;
        std::cout << "  Position: [" << initial_guess[0] << ", " 
                  << initial_guess[1] << ", " << initial_guess[2] << "]" << std::endl;
        std::cout << "  Threshold: " << outlier_threshold * 1000 << " mm" << std::endl;
    }
    
    std::cout << "========================================\n" << std::endl;

    CharucoProcessor processor(input_dir, tcp_poses_file, calib_file, 
                               initial_guess, use_initial_guess, outlier_threshold);
    processor.processAllImages();

    return 0;
}