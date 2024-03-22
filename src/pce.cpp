#include <pce.h>
#include <pce_params.h>

pce::pce()
{
    mytimer = 0.0;
    setParams();

    std::filesystem::path currentPath = std::filesystem::current_path();
    currentPath = currentPath.parent_path();
    currentPath /= "bags/pce_bag.csv";
    datafile_pcebag = currentPath.c_str(); // datafile to save data
    fid_pcebag = fopen(datafile_pcebag,"w"); // open it for write

    std::filesystem::path otherPath = std::filesystem::current_path();
    otherPath = otherPath.parent_path();
    otherPath /= "bags/pce_probs.csv";
    datafile_pceprob = otherPath.c_str(); // datafile to save data
    fid_pceprob = fopen(datafile_pceprob,"w"); // open it for write
}
pce::pce(std::string simulation)
{
    mytimer = 0.0;
    setParams();

    std::filesystem::path currentPath = std::filesystem::current_path();
    currentPath = currentPath.parent_path();
    currentPath = currentPath.parent_path();
    currentPath /= "create_plots/pce_bag.csv";
    datafile_pcebag = currentPath.c_str(); // datafile to save data
    fid_pcebag = fopen(datafile_pcebag,"w"); // open it for write

}
pce::~pce()
{
    // fclose(fid_pcebag);
    fclose(fid_pceprob);

}
// Call this for times = batch_size*dt seconds
void pce::init_things(Eigen::VectorXd imu_msg)
{
    if(data_batch_ax.size() < batch_size)
    {
        initialize_batch(imu_msg); // keep push back current measurements from sensors
    }

}
void pce::setParams()
{
    batch_size   = param_batch_size;
    eval_samples = param_eval_samples;

    ax_bias = param_ax_bias;
    ay_bias = param_ay_bias;
    az_bias = param_az_bias;
    wx_bias = param_wx_bias;
    wy_bias = param_wy_bias;
    wz_bias = param_wz_bias;

    ax_std = param_ax_std;
    ay_std = param_ay_std;
    az_std = param_az_std;
    wx_std = param_wx_std;
    wy_std = param_wy_std;
    wz_std = param_wz_std;

    thres_ax = param_thres_ax;
    thres_ay = param_thres_ay;
    thres_az = param_thres_az;
    thres_wx = param_thres_wx;
    thres_wy = param_thres_wy;
    thres_wz = param_thres_wz;

    Fz_thresshold = params_Fz_thresshold;

}
void pce::initialize_batch(Eigen::VectorXd imu_msg)
{

    data_batch_ax.push_back(imu_msg(0) - ax_bias);
    data_batch_ay.push_back(imu_msg(1) - ay_bias);
    data_batch_az.push_back(imu_msg(2) - az_bias);
    data_batch_wx.push_back(imu_msg(3) - wx_bias);
    data_batch_wy.push_back(imu_msg(4) - wy_bias);
    data_batch_wz.push_back(imu_msg(5) - wz_bias);

}
double pce::stable_contact_detection(Eigen::VectorXd imu_msg)
{
    ax = imu_msg(0) - ax_bias;
    ay = imu_msg(1) - ay_bias;
    az = imu_msg(2) - az_bias;

    wx = imu_msg(3) - wx_bias;
    wy = imu_msg(4) - wy_bias;
    wz = imu_msg(5) - wz_bias;

    //  data_batch -> Batch of data to made predictions on

    //Delete first element
    data_batch_ax.erase(data_batch_ax.begin());
    data_batch_ay.erase(data_batch_ay.begin());
    data_batch_az.erase(data_batch_az.begin());

    data_batch_wx.erase(data_batch_wx.begin());
    data_batch_wy.erase(data_batch_wy.begin());
    data_batch_wz.erase(data_batch_wz.begin());

    //Append the new neasurement
    data_batch_ax.push_back(ax); 
    data_batch_ay.push_back(ay); 
    data_batch_az.push_back(az); 

    data_batch_wx.push_back(wx); 
    data_batch_wy.push_back(wy); 
    data_batch_wz.push_back(wz); 


    try
    {
        //kde 1D
        kdepp::Kde1d<double> kernel_ax(data_batch_ax);
        kdepp::Kde1d<double> kernel_ay(data_batch_ay);
        kdepp::Kde1d<double> kernel_az(data_batch_az);

        kdepp::Kde1d<double> kernel_wx(data_batch_wx);
        kdepp::Kde1d<double> kernel_wy(data_batch_wy);
        kdepp::Kde1d<double> kernel_wz(data_batch_wz);
        // Set bandwith as std
        kernel_ax.set_bandwidth(ax_std);
        kernel_ay.set_bandwidth(ay_std);
        kernel_az.set_bandwidth(az_std);
        kernel_wx.set_bandwidth(wx_std);
        kernel_wy.set_bandwidth(wy_std);
        kernel_wz.set_bandwidth(wz_std);

        // auto result_ax = kernel_ax.eval(0.0);
        // std::cout << result_ax << std::endl;

        prob_ax = get_axis_probability(-thres_ax,thres_ax,kernel_ax);
        prob_ay = get_axis_probability(-thres_ay,thres_ay,kernel_ay);
        prob_az = get_axis_probability(-thres_az,thres_az,kernel_az);

        prob_wx = get_axis_probability(-thres_wx,thres_wx,kernel_wx);
        prob_wy = get_axis_probability(-thres_wy,thres_wy,kernel_wy);
        prob_wz = get_axis_probability(-thres_wz,thres_wz,kernel_wz);

    }
    catch(const std::invalid_argument& e)
    {  
        std::cout<<"MY CATCH"<<std::endl;
        return 0.0;
    }

    return prob_ax*prob_ay*prob_wz*prob_wx*prob_wy*prob_az ;

}
void pce::one_loop(Eigen::VectorXd imu_msg)
{
    mytimer += 2;
    probability = stable_contact_detection(imu_msg);
    // std::cout<<probability<<std::endl;
}
void pce::one_loop( double Fz, Eigen::VectorXd imu_msg)
{
    mytimer += 2;
    if(Fz > Fz_thresshold)
        probability = stable_contact_detection(imu_msg);
    else 
        probability = 0;  // no contact

    std::cout<<probability<<std::endl;
}

double pce::get_axis_probability(double start_value, double end_value,kdepp::Kde1d<double> kd)
{
    double step = (end_value - start_value)/(eval_samples - 1);
    double area = 0.0;
    for(double i = start_value; i < end_value; i+=step)
    {
        area += kd.eval(i)*step;
    }

    return area;
}
void pce::store_data(Eigen::VectorXd imu_msg)
{
    //Append the new neasurement
    data_batch_ax.push_back(imu_msg(0)); 
    data_batch_ay.push_back(imu_msg(1)); 
    data_batch_az.push_back(imu_msg(2)); 

    data_batch_wx.push_back(imu_msg(3)); 
    data_batch_wy.push_back(imu_msg(4)); 
    data_batch_wz.push_back(imu_msg(5)); 
}
void pce::compute_mean_std()
{
    using namespace boost::accumulators;
    using namespace std;
    using namespace placeholders;

    // Mean and std for Ax
    accumulator_set<double, stats<tag::variance> > acc_ax;
    for_each(data_batch_ax.begin(), data_batch_ax.end(), bind<void>(ref(acc_ax), _1));
    double mean_ax = mean(acc_ax);
    double std_ax =  sqrt(variance(acc_ax)) ;

    // Mean and std for Ay
    accumulator_set<double, stats<tag::variance> > acc_ay;
    for_each(data_batch_ay.begin(), data_batch_ay.end(), bind<void>(ref(acc_ay), _1));
    double mean_ay = mean(acc_ay);
    double std_ay =  sqrt(variance(acc_ay)) ;

    // Mean and std for Ay
    accumulator_set<double, stats<tag::variance> > acc_az;
    for_each(data_batch_az.begin(), data_batch_az.end(), bind<void>(ref(acc_az), _1));
    double mean_az = mean(acc_az);
    double std_az =  sqrt(variance(acc_az)) ;

    // Mean and std for Wx
    accumulator_set<double, stats<tag::variance> > acc_wx;
    for_each(data_batch_wx.begin(), data_batch_wx.end(), bind<void>(ref(acc_wx), _1));
    double mean_wx = mean(acc_wx);
    double std_wx =  sqrt(variance(acc_wx)) ;

    // Mean and std for Wy
    accumulator_set<double, stats<tag::variance> > acc_wy;
    for_each(data_batch_wy.begin(), data_batch_wy.end(), bind<void>(ref(acc_wy), _1));
    double mean_wy = mean(acc_wy);
    double std_wy =  sqrt(variance(acc_wy)) ;

    // Mean and std for Wy
    accumulator_set<double, stats<tag::variance> > acc_wz;
    for_each(data_batch_wz.begin(), data_batch_wz.end(), bind<void>(ref(acc_wz), _1));
    double mean_wz = mean(acc_wz);
    double std_wz =  sqrt(variance(acc_wz)) ;
        
    //set Params
    ax_bias = mean_ax;
    ay_bias = mean_ay;
    az_bias = mean_az;
    wx_bias = mean_wx;
    wy_bias = mean_wy;
    wz_bias = mean_wz;

    ax_std = std_ax;
    ay_std = std_ay;
    az_std = std_az;
    wx_std = std_wx;
    wy_std = std_wy;
    wz_std = std_wz;

    // std::cout<<"Size "<< data_batch_ax.size()<<std::endl;

    std::cout<<"Set the following in pce_params.h "<<std::endl;

    std::cout<<"param_ax_bias = "<<ax_bias <<std::endl;
    std::cout<<"param_ay_bias = "<<ay_bias <<std::endl;
    std::cout<<"param_az_bias = "<<az_bias <<std::endl;
    std::cout<<"param_wx_bias = "<<wx_bias <<std::endl;
    std::cout<<"param_wy_bias = "<<wy_bias <<std::endl;
    std::cout<<"param_wz_bias = "<<wz_bias <<std::endl; 


    std::cout<<"param_ax_std = "<<ax_std <<std::endl;
    std::cout<<"param_ay_std = "<<ay_std <<std::endl;
    std::cout<<"param_az_std = "<<az_std <<std::endl;
    std::cout<<"param_wx_std = "<<wx_std <<std::endl;
    std::cout<<"param_wy_std = "<<wy_std <<std::endl;
    std::cout<<"param_wz_std = "<<wz_std <<std::endl;

    data_batch_ax.clear(); 
    data_batch_ay.clear(); 
    data_batch_az.clear(); 
    data_batch_wx.clear(); 
    data_batch_wy.clear(); 
    data_batch_wz.clear(); 


}
void pce::save_csv()
{
    for(int i = 0 ; i< data_batch_ax.size() ; i++ )
    {
        fprintf(fid_pcebag, "%f %f %f %f %f %f", data_batch_ax[i], data_batch_ay[i], data_batch_az[i], data_batch_wx[i],data_batch_wy[i],data_batch_wz[i] );
        fprintf(fid_pcebag,"\n");
    }

}

void pce::save_probability()
{

    fprintf(fid_pceprob, "%d %f", mytimer, probability);
    fprintf(fid_pceprob,"\n");
}
