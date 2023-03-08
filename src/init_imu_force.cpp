#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <iostream>
#include <unitree_legged_msgs/HighState.h>

using namespace boost::accumulators;
using namespace std;

std::vector< double > arr_ax, arr_ay, arr_az;
std::vector< double > arr_wx, arr_wy, arr_wz;
std::vector< double > arr_f;

void initimuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
     
    // linear_acceleration
    arr_ax.push_back(msg->linear_acceleration.x);
    arr_ay.push_back(msg->linear_acceleration.y);
    arr_az.push_back(msg->linear_acceleration.z);

    // angular_velocity
    arr_wx.push_back(msg->angular_velocity.x);
    arr_wy.push_back(msg->angular_velocity.y);
    arr_wz.push_back(msg->angular_velocity.z);
    

}

void initforceCallback(const unitree_legged_msgs::HighState::ConstPtr &msg)
{
     
    // force 
    arr_f.push_back(msg->footForce[2]); // TODO which foot

}

// void calc_mean_std()
// {
//     // mean of angular_velocity vectors
//     double mean_wx = std::accumulate(arr_wx.begin(), arr_wx.end(), decltype(arr_wx)::value_type(0));
//     mean_wx /= num_collect;
//     double mean_wy = std::accumulate(arr_wy.begin(), arr_wy.end(), decltype(arr_wy)::value_type(0));
//     mean_wy /= num_collect;
//     double mean_wz = std::accumulate(arr_wz.begin(), arr_wz.end(), decltype(arr_wz)::value_type(0));
//     mean_wz /= num_collect;

//     // mean of linear_acceleration vectors
//     double mean_ax = std::accumulate(arr_ax.begin(), arr_ax.end(), decltype(arr_ax)::value_type(0));
//     mean_ax /= num_collect;
//     double mean_ay = std::accumulate(arr_ay.begin(), arr_ay.end(), decltype(arr_ay)::value_type(0));
//     mean_ay /= num_collect;
//     double mean_az = std::accumulate(arr_az.begin(), arr_az.end(), decltype(arr_az)::value_type(0));
//     mean_az /= num_collect;
// }

int main(int argc, char **argv)
{
    ros::init(argc, argv, "init_imu_things");

    std::cout << "This will store data for Imu sensor Bias at imuBias.txt" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();
    
    printf("Wait for 4sec...\t");
    
    ros::NodeHandle nh;

    ros::Rate loop_rate(250);

    ros::Subscriber imu_sub = nh.subscribe("imu", 1, initimuCallback);
    ros::Subscriber force_sub = nh.subscribe("high_state", 1, initforceCallback);

    long num_collect  = 0;

    while (ros::ok() and num_collect < 1000)
    {
        num_collect += 1;

        ros::spinOnce();
        loop_rate.sleep();
    }


    
    // Mean and std for Ax
    accumulator_set<double, stats<tag::variance> > acc_ax;
    for_each(arr_ax.begin(), arr_ax.end(), bind<void>(ref(acc_ax), _1));
    double mean_ax = mean(acc_ax);
    double std_ax =  sqrt(variance(acc_ax)) ;

    // Mean and std for Ay
    accumulator_set<double, stats<tag::variance> > acc_ay;
    for_each(arr_ay.begin(), arr_ay.end(), bind<void>(ref(acc_ay), _1));
    double mean_ay = mean(acc_ay);
    double std_ay =  sqrt(variance(acc_ay)) ;

    // Mean and std for Ay
    accumulator_set<double, stats<tag::variance> > acc_az;
    for_each(arr_az.begin(), arr_az.end(), bind<void>(ref(acc_az), _1));
    double mean_az = mean(acc_az);
    double std_az =  sqrt(variance(acc_az)) ;

    // Mean for F
    accumulator_set<double, stats<tag::variance> > acc_f;
    for_each(arr_f.begin(), arr_f.end(), bind<void>(ref(acc_f), _1));
    double mean_f = mean(acc_f);

    // Mean and std for Wx
    accumulator_set<double, stats<tag::variance> > acc_wx;
    for_each(arr_wx.begin(), arr_wx.end(), bind<void>(ref(acc_wx), _1));
    double mean_wx = mean(acc_wx);
    double std_wx =  sqrt(variance(acc_wx)) ;

    // Mean and std for Wy
    accumulator_set<double, stats<tag::variance> > acc_wy;
    for_each(arr_wy.begin(), arr_wy.end(), bind<void>(ref(acc_wy), _1));
    double mean_wy = mean(acc_wy);
    double std_wy =  sqrt(variance(acc_wy)) ;

    // Mean and std for Wy
    accumulator_set<double, stats<tag::variance> > acc_wz;
    for_each(arr_wz.begin(), arr_wz.end(), bind<void>(ref(acc_wz), _1));
    double mean_wz = mean(acc_wz);
    double std_wz =  sqrt(variance(acc_wz)) ;
    


    ofstream myfile;    // file for imu sensor things
    myfile.open ("/home/despargy/go1_ws/src/go1_motion/src/exe/imuBias.txt");

    myfile << mean_ax << ',' << mean_ay << ',' << mean_az << ',';
    myfile << mean_wx << ',' << mean_wy << ','<< mean_wz << ',';
    
    // TODO fz_bias
    myfile << mean_f  << ',';

    myfile << std_ax << ',' << std_ay << ',' << std_az << ',';
    myfile << std_wx << ',' << std_wy << ',' << std_wz ;

    // close file
    myfile.close();

    printf("Bias and std at your service!\t");

    return 0;
}