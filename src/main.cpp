#include <pce.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>


//Open and read ofline data from cvs, return as (vector of (vector of doubles))
std::vector<std::vector<double>> readCSV(const std::string& filename)
{
    std::ifstream file(filename);
    std::vector<std::vector<double>> data;

    if(!file.is_open())
    {
        std::cerr<<"Unable to open the file: "<<filename<<std::endl;
        return data;
    }
    else
    {
        std::string meassurament; // read as string pass it as double
        std::vector<double> row;

        std::string line; // read as string each csv row
        while(std::getline(file, line))
        {
            try{
                row.clear();
                std::stringstream stream(line);
                
                while(std::getline(stream, meassurament, ','))
                {
                    double val = std::stod(meassurament);
                    row.push_back(val);
                }
                // pass another message row to data
                data.push_back(row);
            }catch (const std::invalid_argument& e) {
                  std::cerr << "Invalid argument: " << meassurament << std::endl;
            }
        }
        file.close();
    }
    return data;
}



int main()
{

    // Read ofline data from csv
    std::vector<std::vector<double>> offline_data = readCSV("../offline/data/GO1_normal_surface.csv");
    // Print the ofline data from csv
    // for(const auto& row: ofline_data)
    // {
    //     std::cout<< "msg (Fz ax, ay, az, wx, wy, wz) : " <<std::endl;
    //     for(const auto& val: row)
    //     {
    //         std::cout<< val <<std::endl;
    //     }
    //     std::cout<< "  " <<std::endl;

    // } 

    // PCE
    pce pce_obj;

    double Fz;
    Eigen::VectorXd imu_msg;
    imu_msg.resize(6);

    bool FIRST_TIME_CALC_PARAMS = false;

/*************************************/
/* Replace the printed values in pce_params.h file */
/* After replace it se FIRST_TIME_CALC_PARAMS to false */
    if(FIRST_TIME_CALC_PARAMS)
    {
        int during_stance=1000;
        for( int i=0; i < during_stance; i++)
        {
            imu_msg(0) = offline_data[i][1];
            imu_msg(1) = offline_data[i][2];
            imu_msg(2) = offline_data[i][3];
            imu_msg(3) = offline_data[i][4];
            imu_msg(4) = offline_data[i][5];
            imu_msg(5) = offline_data[i][6]; 
            pce_obj.store_data(imu_msg);
        }
        pce_obj.compute_mean_std();
    }
/*************************************/

/*************************************/
/* Main flow for PCE */
    else 
    {
        // Push back 'batch_size' meassurements to start PCE
        for( int i=0; i < pce_obj.batch_size; i++)
        {
            // Fz = ofline_data[i][0];
            imu_msg(0) = offline_data[i][1];
            imu_msg(1) = offline_data[i][2];
            imu_msg(2) = offline_data[i][3];
            imu_msg(3) = offline_data[i][4];
            imu_msg(4) = offline_data[i][5];
            imu_msg(5) = offline_data[i][6]; 

            pce_obj.init_things(imu_msg);
        }
        
        /* For real time data, feed per dt 
            the "one_loop" function with imu row msg,
                instead of the "for loop"       */
        for(const auto& msg: offline_data)  // May not read the first 'pce_obj.batch_size' msgs
        {
            //update Fz with 'current' msg
            Fz = msg[0];

            //update imu_msg with 'current' msg
            imu_msg(0) = msg[1];
            imu_msg(1) = msg[2];
            imu_msg(2) = msg[3];
            imu_msg(3) = msg[4];
            imu_msg(4) = msg[5];
            imu_msg(5) = msg[6]; 
    
            // Choose betweem 'one_loop' to compute probability if Fz is above the Fz_thresshold
            pce_obj.one_loop(imu_msg);
            // pce_obj.one_loop(Fz, imu_msg);

            pce_obj.save_probability();
        }
    }
    /* Notes:
        
        -First time to compute bias and std: keep the robot standup
            for (some time during stance)
                $store_data();
            then call
            $compute_mean_std();

        -Main flow
            for some time = batch_size*dt
                $init_things()
            while calc_probability
                $update: imu_msg, Fz
                $one_loop(imu_msg) or one_loop(Fz, imu_msg)
                $save_probability() in csv
    
    */

    return 0;
}