#ifndef _PCE_PARAMS_H_
#define _PCE_PARAMS_H_

#define param_batch_size 50;
#define param_eval_samples 500;

/* Replace the output of pce when FIRST_TIME_CALC_PARAMS = true */
#define param_ax_bias -0.00901782;
#define param_ay_bias -0.965768;
#define param_az_bias 0.257405;
#define param_wx_bias 0.848511;
#define param_wy_bias 1.81409;
#define param_wz_bias 5.07727;

#define param_ax_std 0.662803;
#define param_ay_std 0.797816;
#define param_az_std 0.452588;
#define param_wx_std 23.2532;
#define param_wy_std 40.2488;
#define param_wz_std 123.126;

/* Tune thresshold parameters */
#define param_thres_ax  100;     
#define param_thres_ay  100;
#define param_thres_az  100;
#define param_thres_wx  100;
#define param_thres_wy  100;
#define param_thres_wz  100;
#define params_Fz_thresshold 0; // If needed

#endif