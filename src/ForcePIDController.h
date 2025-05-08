#ifndef ForcePIDController_H_
#define ForcePIDController_H_

#include <algorithm>  // for std::clamp

class ForcePIDController {
public:
    ForcePIDController(float Kp_, float Ki_, float Kd_, float dt_, float vel_min_, float vel_max_)
        : Kp(Kp_), Ki(Ki_), Kd(Kd_), dt(dt_), vel_min(vel_min_), vel_max(vel_max_),
          integral(0.0), prev_error(0.0)
    {}

    float computeVelocity(float ct, float fz_measured) {
        // Time-varying target force
        float fz_target;
        fz_target = -270.0f;  // Default target force
        /* if (ct < 30.0f)
            fz_target = -40.0f;
        else if (ct > 40.0f)
            fz_target = -270.0f;
        else
            fz_target = -1.*(23. * ct - 650.); */

        // Error signal
        float error = fz_target - fz_measured;

        // PID terms
        integral += error * dt;
        float derivative = (error - prev_error) / dt;

        float output = Kp * error + Ki * integral + Kd * derivative;

        // Anti-windup: clamp integral to prevent runaway
        const float integral_max = 1000.0f;  // Adjust based on system
        integral = std::clamp(integral, -integral_max, integral_max);

        // Save error for next cycle
        prev_error = error;

        // Clamp velocity to safe bounds
        float velocity_command = std::clamp(output, vel_min, vel_max);

        return velocity_command;
    }

private:
    float Kp, Ki, Kd;
    float dt;
    float vel_min, vel_max;
    float integral;
    float prev_error;
};

#endif /* ACTIONS_GPU_H_ */
