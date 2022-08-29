#include "ros/ros.h"


#include <iostream>
#include <fstream>
#include <vector>



int main(int argc,char **argv)
{

    ros::init(argc, argv, "ContactEstimation");
    ros::NodeHandle nh;

    std::ifstream file("../data/atlas_1000hz_01ground_3steps.csv");
    std::vector<double> data_array;
    double data_point;

    while (file >> data_point){
        data_array.push_back(data_point);
    }

    for (int i = 0 ; i < data_array.size()/7 ; i++){
        
    }

    return 0;
}