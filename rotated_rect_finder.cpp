#include "aricc_2d_vision/rotated_rect_finder.h"
#define PI 3.141592


namespace aricc_2d_vision{

  void RotatedRectFinder::onInit(){
    ConnectionBasedNodelet::onInit();
    pnh_->param("approximate_sync", approximate_sync_, false);
    pnh_->param("debug", debug_, false);
    pnh_->param("detect_color", detect_color_, false);
    pnh_->param("detect_density", detect_density_, false);
    pnh_->param("z", z_, 1.0);
    srv_ = boost::make_shared <dynamic_reconfigure::Server<Config> > (*pnh_);
    dynamic_reconfigure::Server<Config>::CallbackType f =
      boost::bind (
        &RotatedRectFinder::configCallback, this, _1, _2);
    srv_->setCallback (f);
    if(debug_){
      pub_debug_image_ = advertise<sensor_msgs::Image>(*pnh_, "debug", 1);
    }
    pub_rects_ = advertise<aricc_vision_msgs::RotatedRectArray>(*pnh_, "output/rects", 1);
    pub_rects_p_ = advertise<aricc_vision_msgs::RotatedRectArray>(*pnh_, "output/projected_rects", 1);
  }

  void RotatedRectFinder::subscribe(){
    sub_contour_.subscribe(*pnh_,"input/contour", 1);
    sub_rgb_image_.subscribe(*pnh_,"input/rgb_image", 1);
    sub_threshold_image_.subscribe(*pnh_,"input/threshold_image", 1);
    sub_image_info_.subscribe(*pnh_,"input/image_info", 1);
    if (approximate_sync_) {
      async_ = boost::make_shared<message_filters::Synchronizer<ApproxSyncPolicy> >(100);
      async_->connectInput(sub_contour_, sub_rgb_image_, sub_threshold_image_, sub_image_info_);
      async_->registerCallback(boost::bind(&RotatedRectFinder::execute, this, _1, _2, _3, _4));
    }
    else {
      sync_ = boost::make_shared<message_filters::Synchronizer<SyncPolicy> >(100);
      sync_->connectInput(sub_contour_, sub_rgb_image_, sub_threshold_image_,sub_image_info_);
      sync_->registerCallback(boost::bind(&RotatedRectFinder::execute, this, _1, _2, _3, _4));
    }
  }

  void RotatedRectFinder::unsubscribe(){
    sub_contour_.unsubscribe();
    sub_image_info_.unsubscribe();
    sub_rgb_image_.unsubscribe();
    sub_threshold_image_.unsubscribe();
  }

  void RotatedRectFinder::configCallback(
    Config &config, uint32_t level){
    boost::mutex::scoped_lock lock(mutex_);
    debug_ = config.debug;
    result_= config.result;
    ruler_ = config.ruler;
    z_     = config.z;
  }


  void RotatedRectFinder::pubDebug( cv::Mat& src, std_msgs::Header header ){
      if(result_) drawResult(src);
      if(ruler_) drawRuler(src);
      pub_debug_image_.publish(cv_bridge::CvImage(
                               header,
                               sensor_msgs::image_encodings::BGR8,
                               src).toImageMsg());
      /*
      pub_debug_image_.publish(cv_bridge::CvImage(
                               header,
                               sensor_msgs::image_encodings::MONO8,
                               src).toImageMsg());*/
  }

  double RotatedRectFinder::findDensity(std::vector<cv::Point> contour, cv::Mat image){
    cv::Rect rect = cv::boundingRect( cv::Mat(contour));
    int counter = 0;
    double color_sum = 0;
    for(unsigned int i = rect.y; i < (rect.y+rect.height); ++i ){
      unsigned char *data = image.ptr<unsigned char>(i) + rect.x;
      for(unsigned int j = rect.x; j < (rect.x+rect.width); ++j){
        unsigned char *color = data++;
        int res = pointPolygonTest(contour, cv::Point(j,i),false);
        if(res == 1){
          counter++;
          color_sum += (unsigned int)*color;
        }
      }
    }
    return color_sum/counter;
  }

  void RotatedRectFinder::drawRect(
    std::vector<cv::Point> contour, cv::Mat& image){
    cv::Rect rect = cv::boundingRect( cv::Mat(contour));
    for(unsigned int i = rect.y; i < (rect.y+rect.height); ++i ){
      unsigned char *data = image.ptr<unsigned char>(i) + rect.x;
      for(unsigned int j = rect.x; j < (rect.x+rect.width); ++j){
        int res = pointPolygonTest(contour, cv::Point(j,i),false);
        if(res == 1) *data = 100;
        data++;
      }
    }
  }

  double RotatedRectFinder::getDist( geometry_msgs::Point p1,
                                     geometry_msgs::Point p2){
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    double dist = sqrt(dx*dx + dy*dy + dz*dz);
    return dist;
  }

  void RotatedRectFinder::getDim(aricc_vision_msgs::RotatedRect& rect){
    geometry_msgs::Point p0 = rect.points[0];
    geometry_msgs::Point p1 = rect.points[1];
    geometry_msgs::Point p2 = rect.points[2];

    double dist_1 = getDist(p0, p1);
    double dist_2 = getDist(p1, p2);
    if(dist_1 <= dist_2) {
      rect.width = dist_1;
      rect.height = dist_2;
      double dx = p2.x - p1.x;
      double dy = p2.y - p1.y;
      rect.angle = atan(dy/dx);
    }
    else{
      rect.width = dist_2;
      rect.height = dist_1;
      double dx = p1.x - p0.x;
      double dy = p1.y - p0.y;
      rect.angle = atan(dy/dx);
    }
    //NODELET_INFO("W:%lf, H:%lf, A:%lf",rect.width, rect.height,rect.angle*180.0/M_PI);
  }

  void RotatedRectFinder::toROSMsg(cv::RotatedRect rect,
                                   double color_h,
                                   double color_s,
                                   double color_v,
                                   double density,
                                   aricc_vision_msgs::RotatedRect& msg){
    msg.center.y = rect.center.x;
    msg.center.x = rect.center.y;
    msg.width    = rect.size.width;
    msg.height   = rect.size.height;
    msg.color_h  = color_h;
    msg.color_s  = color_s;
    msg.color_v  = color_v;
    msg.density  = density;
    cv::Point2f vertices[4];
    rect.points(vertices);
    for(size_t i = 0; i < 4; ++i){
      geometry_msgs::Point p;
      p.y = vertices[i].x; p.x = vertices[i].y;
      msg.points.push_back(p);
    }
  }
    /*OpenCV coordinate ---->x
                        |
                        |
                        v y
      ROS coordinate    ---->y
                        |
                        |
                        v x
    */

  void RotatedRectFinder::project(
    aricc_vision_msgs::RotatedRect rect,
    aricc_vision_msgs::RotatedRect& rect_p ){
    cv::Point3d ray;
    ray = model_.projectPixelTo3dRay(
          cv::Point2d(rect.center.y, rect.center.x));
    double alpha = z_ / ray.z;
    //ROS_INFO("%.3lf",z_);
    rect_p.center.x = -ray.y*alpha;
    rect_p.center.y = -ray.x*alpha;
    rect_p.center.z = ray.z*alpha;
    //rect_p.center.y = ray.x*alpha;
    //rect_p.center.x = ray.y*alpha;
    //rect_p.center.z = ray.z*alpha;
    rect_p.color_h    = rect.color_h;
    rect_p.color_s    = rect.color_s;
    rect_p.color_v    = rect.color_v;
    rect_p.density  = rect.density;
    for(size_t i = 0; i < rect.points.size(); ++i){
      ray = model_.projectPixelTo3dRay(
            cv::Point2d(rect.points.at(i).y, rect.points.at(i).x));
      geometry_msgs::Point p;
      p.y = -ray.x*alpha;
      p.x = -ray.y*alpha;
      p.z = ray.z*alpha;
      rect_p.points.push_back(p);
    }
  }

  void RotatedRectFinder::drawResult(cv::Mat& src){
      cv::RNG rng(12345);
      for(size_t i = 0; i < msg_rects_.rects.size(); ++i){
        cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        cv::Point2f center;
        center.x = msg_rects_.rects.at(i).center.y;
        center.y = msg_rects_.rects.at(i).center.x;
        if(msg_rects_.rects.at(i).color_h != -1 &&
           msg_rects_.rects.at(i).color_s != -1 &&
           msg_rects_.rects.at(i).color_v != -1 ){
          cv::Scalar c = cv::Scalar(
            msg_rects_.rects.at(i).color_h,
            msg_rects_.rects.at(i).color_s,
            msg_rects_.rects.at(i).color_v);
          cv::circle( src, center, 10, c, -1, 8, 0 );
        }
        else
          cv::circle( src, center, 10, color, 1, 8, 0 );
        cv::Point2f p;
        double angle = msg_rects_.rects.at(i).angle*(180/PI);

        //NODELET_INFO("Angle:%lf",angle/M_PI*180.0);
        p.x = center.x + 20*sin(angle);
        p.y = center.y + 20*cos(angle);
        cv::line( src, center, p, color, 1, 8 );
        //Draw rotated rect
        cv::Point2f vertices[4];
        for(size_t j = 0; j < 4; ++j){
          vertices[j].y = msg_rects_.rects.at(i).points[j].x;
          vertices[j].x = msg_rects_.rects.at(i).points[j].y;
        }

        for(size_t j = 0; j < 4; ++j){
          line( src, vertices[j], vertices[(j+1)%4], color);
        }
        cv::putText(src, std::to_string(angle), cv::Point(center.x,center.y), cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(150,0,190),2,false);
      }

  }

  void RotatedRectFinder::drawRuler(cv::Mat& src){
    //std::vector<cv::Point3d> points_3d;
    //std::vector<cv::Point2d> points_2d;
    double step = 0.01;
    cv::Scalar color = cv::Scalar( 255, 0, 0);
    for(size_t i = 0; i < 21; ++i){
      cv::Point3d p_3d_f, p_3d_s;
      cv::Point2d p_2d_f, p_2d_s;
      p_3d_f.x = -0.05 + step*i;
      p_3d_f.y = -0.05;
      p_3d_f.z = z_;
      p_3d_s.x = -0.05 + step*i;
      p_3d_s.y = 0.05;
      p_3d_s.z = z_;
      p_2d_f = model_.project3dToPixel(p_3d_f);
      p_2d_s = model_.project3dToPixel(p_3d_s);
      cv::line(src, p_2d_f, p_2d_s, color, 1, 8 );
    }
    for(size_t i = 0; i < 21; ++i){
      cv::Point3d p_3d_f, p_3d_s;
      cv::Point2d p_2d_f, p_2d_s;
      p_3d_f.x = -0.05;
      p_3d_f.y = -0.05 + step*i;
      p_3d_f.z = z_;
      p_3d_s.x = 0.05;
      p_3d_s.y = -0.05 + step*i;
      p_3d_s.z = z_;
      p_2d_f = model_.project3dToPixel(p_3d_f);
      p_2d_s = model_.project3dToPixel(p_3d_s);
      cv::line(src, p_2d_f, p_2d_s, color, 1, 8 );
    }
    cv::Point3d p_3d_c;
    cv::Point2d p_2d_c;
    p_3d_c.x = 0.0;
    p_3d_c.y = 0.0;
    p_3d_c.z = z_;
    p_2d_c = model_.project3dToPixel(p_3d_c);
    cv::circle(src, p_2d_c, 3, color, 1, 8, 0);
  }

  void RotatedRectFinder::execute(
    const aricc_vision_msgs::ContourArray::ConstPtr& msg,
    const sensor_msgs::Image::ConstPtr& image_msg,
    const sensor_msgs::Image::ConstPtr& threshold_image_msg,
    const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg){

    boost::mutex::scoped_lock lock(mutex_);
    std::vector< std::vector<cv::Point> > contours;
    contours.clear();
    model_.fromCameraInfo(camera_info_msg);

    //Convert input image to HSV colour space
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(
      image_msg, image_msg->encoding);
    cv::Mat rgb_image = cv_ptr->image;
    cv::Mat hsv_image;

    if(image_msg->encoding == sensor_msgs::image_encodings::MONO8) {
      cv::Mat tmp_image;
      cv::cvtColor(rgb_image, tmp_image, CV_GRAY2RGB);
      cv::cvtColor(tmp_image, hsv_image, CV_RGB2HSV);
    }

    else if (image_msg->encoding == sensor_msgs::image_encodings::BGR8) {
      cv::cvtColor(rgb_image, hsv_image, CV_BGR2HSV);
    }
    else if (image_msg->encoding == sensor_msgs::image_encodings::RGB8) {
      cv::cvtColor(rgb_image, hsv_image, CV_RGB2HSV);
    }
    else if (image_msg->encoding == sensor_msgs::image_encodings::BGRA8 || image_msg->encoding == sensor_msgs::image_encodings::BGRA16) {
      cv::Mat tmp_image;
      cv::cvtColor(rgb_image, tmp_image, CV_BGRA2BGR);
      cv::cvtColor(tmp_image, hsv_image, CV_BGR2HSV);
    }
    else if (image_msg->encoding == sensor_msgs::image_encodings::RGBA8 || image_msg->encoding == sensor_msgs::image_encodings::RGBA16) {
      cv::Mat tmp_image;
      cv::cvtColor(rgb_image, tmp_image, CV_RGBA2BGR);
      cv::cvtColor(tmp_image, hsv_image, CV_BGR2HSV);
    }
    else {
      NODELET_ERROR("unsupported format to HSV: %s",
        image_msg->encoding.c_str());
      return;
    }

    std::vector<cv::Mat> hsv_planes;
    hsv_planes.clear();
    cv::split(hsv_image, hsv_planes);
    cv::Mat h_image = hsv_planes[0];
    cv::Mat s_image = hsv_planes[1];
    cv::Mat v_image = hsv_planes[2];

    img_width_  = hsv_image.cols;
    img_height_ = hsv_image.rows;

    cv::Mat input_threshold = cv_bridge::toCvCopy(
      threshold_image_msg, sensor_msgs::image_encodings::MONO8)->image;

    //Convert ROS msg to opencvn contour
    std::vector<cv::Point> contour;
    for(size_t i = 0; i < msg->contours.size(); ++i){
      contour.clear();
      for(size_t j = 0; j < msg->contours.at(i).points.size(); ++j){
        cv::Point p;
        p.x = msg->contours.at(i).points.at(j).x;
        p.y = msg->contours.at(i).points.at(j).y;
        contour.push_back(p);
      }
      contours.push_back(contour);
    }
    msg_rects_.rects.clear();
    msg_rects_p_.rects.clear();
    std::vector <std::vector<cv::Point> >::iterator it = contours.begin();
    for( ; it != contours.end(); ){
      aricc_vision_msgs::RotatedRect msg_rect;
      aricc_vision_msgs::RotatedRect msg_rect_p;
      cv::RotatedRect rect = cv::minAreaRect(cv::Mat(*it));
      double color_h = -1;
      double color_s = -1;
      double color_v = -1;
      double density = -1;
      //drawRect(*it, h_image);
      if(detect_color_){
        color_h  = findDensity(*it,h_image);
        color_s  = findDensity(*it,s_image);
        color_v  = findDensity(*it,v_image);
      }
      if(detect_density_) density = findDensity(*it,input_threshold);
      // No projected rects
      toROSMsg(rect,color_h,color_s,color_v,density,msg_rect);
      getDim(msg_rect);
      //Projected rects
      project(msg_rect, msg_rect_p);
      getDim(msg_rect_p);
      msg_rects_.rects.push_back(msg_rect);
      msg_rects_p_.rects.push_back(msg_rect_p);
      ++it;
    }
    if(msg_rects_.rects.size() != 0){
      msg_rects_.header = msg->header;
      msg_rects_.header.stamp = ros::Time::now();
      pub_rects_.publish(msg_rects_);
    }
    if(msg_rects_p_.rects.size() != 0){
      msg_rects_p_.header = msg->header;
      msg_rects_p_.header.stamp = ros::Time::now();
      pub_rects_p_.publish(msg_rects_p_);
    }
    if(debug_){
      cv::Mat drawing = cv_bridge::toCvCopy(
        image_msg, sensor_msgs::image_encodings::BGR8)->image;
      pubDebug(drawing, image_msg->header);
      //pubDebug(h_image, color_image_msg->header);
    }
  }
}

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS (aricc_2d_vision::RotatedRectFinder, nodelet::Nodelet);