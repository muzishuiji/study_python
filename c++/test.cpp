

cv::Point2f recursive_bezier(const std::vector<cv::Point2f> &control_points, float t) 
{
    // TODO: Implement de Casteljau's algorithm
    
    if(control_points.size() == 1) return control_points[0];

    std::vector<cv::Point2f> a;
    for(int i = 0;i+1 < control_points.size();i ++) {
        auto p = control_points[i] + t * (control_points[i+1] - control_points[i]);
        a.push_back(p);
    }

    return recursive_bezier(a, t);
}

int main() {

    // 创建一个含有三个控制点的向量
    std::vector<cv::Point2f> control_points = { cv::Point2f(0, 0), cv::Point2f(0, 2), cv::Point2f(2, 2)};
    
    // 参数t，比如0.5
    float t = 0.5;

    // 调用函数并获取结果
    cv::Point2f bezier_point = recursive_bezier(control_points, t);
  
    std::cout << "The point on the Bezier Curve at t = 0.5 is: (" << bezier_point.x << ", " << bezier_point.y << ")" << std::endl;

    return 0;
}

void bezier(const std::vector<cv::Point2f> &control_points, cv::Mat &window) 
{
    // TODO: Iterate through all t = 0 to t = 1 with small steps, and call de Casteljau's 
    // recursive Bezier algorithm.

    double delta = 0.001;
    for(double t = 0;t <= 1;t += delta) {
        auto point = recursive_bezier(control_points, t);
        int w = 1;
        for(int i = -w+1;i <= w;i ++) {
            for(int j = -w+1;j <= w;j ++) {
                int x = point.x + i, y = point.y + j;

                double dist = sqrt(pow(point.x-x,2) + pow(point.y-y,2));
                window.at<cv::Vec3b>(y, x)[1] = std::min(window.at<cv::Vec3b>(y, x)[1] + 255 * std::max(2-exp(dist),0.0), 255.0);

                // auto k = abs(((int)(point.x+1-i))-point.x) * abs(((int)(point.y+1-j))-point.y);
                // window.at<cv::Vec3b>(y, x)[1] = std::min(window.at<cv::Vec3b>(y, x)[1] + 255 * k, 255.0f);
            }
        }
    }

}