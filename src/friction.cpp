#include <iostream>

#include <ros/ros.h>
#include "geometry_msgs/WrenchStamped.h"
#include  "/home/michael/raisim_ros/devel/include/raisim_ros/StringStamped.h"

raisim_ros::StringStamped contact; 
geometry_msgs::WrenchStamped force;


void contactCallback(const raisim_ros::StringStamped::ConstPtr& msg)
{
  contact = *msg;
}


void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg)
{
  force = *msg;
}

int main(int argc, char **argv)
{
  ROS_INFO("Initializing Friction Estimation Node");
  ros::init(argc, argv, "Friction_Estimator");

  ros::NodeHandle nh;

  ros::Subscriber sub_contact = nh.subscribe("/atlas_raisim_ros/LLeg/contact_status",1000,contactCallback);

  ros::Subscriber sub_forces = nh.subscribe("/atlas_raisim_ros/LLeg/force_torque_states",1000,forceCallback);


  int curr_contact_label;
  int prev_contact_label;
  while (ros::ok()){
  
    // std::cout << contact.data << "\n";
    // switch (contact.data){
    //   case "stable_contact":
    //     curr_contact_label = 0;
    // }
    // std::cout << contact.data << '\n';
    if (contact.data == "stable_contact"){
      curr_contact_label = 0;
    }else if (contact.data == "slip"){
      curr_contact_label = 1;
    }else{
      curr_contact_label = 2;
    }

    if (curr_contact_label == 1 && prev_contact_label == 0){
      auto Fx = force.wrench.force.x;
      auto Fy = force.wrench.force.y;    
      auto Fz = force.wrench.force.z;
      std::cout << "mu = " << std::pow(Fx*Fx + Fy*Fy,0.5)/Fz << '\n';
    }

    prev_contact_label = curr_contact_label;
    ros::spinOnce();

  }



  return 0;
}
