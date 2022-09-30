#include <ros/ros.h>
#include <unitree_legged_msgs/HighCmd.h>
#include <unitree_legged_msgs/HighState.h>
#include "/home/despargy/go1_ws/src/unitree_legged_sdk/include/unitree_legged_sdk/unitree_legged_sdk.h"
#include "convert.h"
#include <std_msgs/Float32.h>
#include <deque>
#include <iostream> 
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

using namespace UNITREE_LEGGED_SDK;
using namespace boost::accumulators;


ros::Publisher pub ;
ros::Subscriber slip_sub ;
ros::Subscriber high_sub ;

// Define them 
int SIZEQ = 70;
double Thres_Prob = 0.7 ; 
int TIMES_OF_STABILITY = 4000;

std::deque<double> mydeque;
long stable_count = 0;
int STATE_CASE = 0;
double PATAEI;

void slipDetectCallback(const std_msgs::Float32::ConstPtr &msg)
{
    // printf("slipDetectCallback is running!\t%ld\n", stable_count++);
    stable_count += 1;

    if (PATAEI > 65)
    {
        // add new value, pop the latest FIFO
        mydeque.push_back (msg->data);
        mydeque.pop_front();

        // calc mean of deque
        accumulator_set<double, stats<tag::variance> > acc_Q;
        for_each(mydeque.begin(), mydeque.end(), bind<void>(ref(acc_Q), _1));

        if (mean(acc_Q) < Thres_Prob )
        {
            // slip detected, change case
            STATE_CASE = 1;

            ROS_INFO("%f", msg->data);

            // reset the stable moments
            stable_count = 0;
        }
    }



}

void highCallback(const unitree_legged_msgs::HighState::ConstPtr &msg)
{
    PATAEI = msg->footForce[1] ;
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "example_walk_without_lcm");

    std::cout << "WARNING: Control level is set to HIGH-level." << std::endl
              << "Make sure the robot is standing on the ground." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    ros::NodeHandle nh;

    ros::Rate loop_rate(500);


    // init queue
    for (int i = 0 ; i < SIZEQ ; i++)   mydeque.push_back(1); // init with stale prob

    pub = nh.advertise<unitree_legged_msgs::HighCmd>("high_cmd", 1000);
    slip_sub = nh.subscribe("contact_probability", 1, slipDetectCallback);
    high_sub = nh.subscribe("high_state", 1, highCallback);

    long motiontime = 0;
    long recoverytime = 0;

    unitree_legged_msgs::HighCmd high_cmd_ros;

    while (ros::ok())
    {

        high_cmd_ros.head[0] = 0xFE;
        high_cmd_ros.head[1] = 0xEF;
        high_cmd_ros.levelFlag = HIGHLEVEL;
        high_cmd_ros.mode = 0;
        high_cmd_ros.gaitType = 0;
        high_cmd_ros.speedLevel = 0;
        high_cmd_ros.footRaiseHeight = 0;
        high_cmd_ros.bodyHeight = 0;
        high_cmd_ros.euler[0] = 0;
        high_cmd_ros.euler[1] = 0;
        high_cmd_ros.euler[2] = 0;
        high_cmd_ros.velocity[0] = 0.0f;
        high_cmd_ros.velocity[1] = 0.0f;
        high_cmd_ros.yawSpeed = 0.0f;
        high_cmd_ros.reserve = 0;

        switch (STATE_CASE)
        {
            case 1: // slip detected 
                /* code */
                recoverytime += 2;
                if (recoverytime > 0 ) //&& recoverytime < 3000
                {
                    // 1. force stand
                    high_cmd_ros.mode = 1;
                }

                break;
            
            default:  // normal mode 0
                motiontime += 2;
                if (motiontime > 0 && motiontime < 3000)
                {
                    high_cmd_ros.mode = 1;
                }
                if (motiontime > 3000 && motiontime < 10000)
                {
                    high_cmd_ros.mode = 2;
                    high_cmd_ros.gaitType = 3;
                    high_cmd_ros.velocity[0] = 0.2f; // -1  ~ +1
                    high_cmd_ros.footRaiseHeight = 0.08;
                    printf("walk\n");
                }
                break;
        }

        pub.publish(high_cmd_ros);

        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}