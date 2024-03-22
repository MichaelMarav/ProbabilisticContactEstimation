#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <bits/stdc++.h>
#include <array>
#include <kde.h>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <vector>
#include <numeric>
#include <functional> 

#ifndef _PCE_H_
#define _PCE_H_


class pce
{
private:
    /* data */
public:

    int batch_size, eval_samples, mytimer;
    double ax_bias, ay_bias, az_bias, wx_bias, wy_bias, wz_bias;
    double ax_std, ay_std, az_std, wx_std, wy_std, wz_std;
    double ax, ay, az, wx, wy, wz;
    double thres_ax, thres_ay, thres_az, thres_wx, thres_wy, thres_wz, Fz_thresshold;
    double probability, prob_ax, prob_ay, prob_az, prob_wx, prob_wy, prob_wz;
    const char* datafile_pcebag; 
    const char* datafile_pceprob;
    FILE* fid_pcebag; 
    FILE* fid_pceprob;
    std::vector<double> data_batch_ax;
    std::vector<double> data_batch_ay;
    std::vector<double> data_batch_az;
    std::vector<double> data_batch_wx;
    std::vector<double> data_batch_wy;
    std::vector<double> data_batch_wz;

    pce();
    pce(std::string simulation);
    ~pce();

    void setParams();
    double stable_contact_detection( Eigen::VectorXd imu_msg);
    void initialize_batch( Eigen::VectorXd imu_msg);
    void one_loop( Eigen::VectorXd imu_msg);
    void one_loop( double Fz, Eigen::VectorXd imu_msg);

    double get_axis_probability(double start_value, double end_value,kdepp::Kde1d<double> kd);
    void init_things( Eigen::VectorXd imu_msg);

    void store_data(Eigen::VectorXd imu_msg);
    void compute_mean_std();
    void save_csv();
    void save_probability();

};




#endif