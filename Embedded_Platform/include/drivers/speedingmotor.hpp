/**
 * Copyright (c) 2019, Bosch Engineering Center Cluj and BFMC organizers
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:

 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.

 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.

 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.

 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
*/

/* Include guard */
#ifndef SPEEDINGMOTOR_HPP
#define SPEEDINGMOTOR_HPP

/* The mbed library */
#include <mbed.h>

namespace drivers
{
    /**
     * @brief Interface to control the brushless motor.
     * 
     */
    class ISpeedingCommand
    {
        public:
            virtual void setSpeed(float f_speed) = 0 ;
            virtual bool inRange(float f_speed) = 0 ;
            virtual void setBrake() = 0 ;
    };

    /**  
     * @brief Speeding motor driver
     * 
     * It is used to control the Brushless motor (more precisely the ESC), which is connected to driving shaft. The reference speed can be accessed through 'setSpeed' method. 
     * 
     */
    class CSpeedingMotor: public ISpeedingCommand
    {
        public:
            /* Constructor */
            CSpeedingMotor(
                PinName     f_pwm_pin,
                float       f_inf_limit,
                float       f_sup_limit
            );
            /* Destructor */
            ~CSpeedingMotor();
            /* Set speed */
            void setSpeed(float f_speed); 
            /* Check speed is in range */
            bool inRange(float f_speed);
            /* Set brake */
            void setBrake(); 

        private:
            /** @brief PWM output pin */
            PwmOut m_pwm_pin;
            /** @brief 0 default */
            uint32_t zero_default = 1500;
            /** @brief 0 default */
            int8_t ms_period = 20;
            /** @brief step_value */
            uint32_t step_value = 5;
            /** @brief Inferior limit */
            const float m_inf_limit;
            /** @brief Superior limit */
            const float m_sup_limit;

            /* interpolate the step value based on the speed value */
            float interpolate(float speed, const float speedValuesP[], const float speedValuesN[], const float stepValues[], int size);

            // Predefined values for steering reference and interpolation
            const float speedValuesP[25] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 26.0, 30.0, 35.0, 40.0, 45.0, 50.0};
            const float speedValuesN[25] = {-4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0, -14.0, -15.0, -16.0, -17.0, -18.0, -19.0, -20.0, -21.0, -22.0, -26.0, -30.0, -35.0, -40.0, -45.0, -50.0};
            const float stepValues[25] = {0.0008346, 0.0006864, 0.0005928, 0.0005226, 0.000468, 0.000429, 0.0003978, 0.0003666, 0.0003354, 0.0003198, 0.0003042, 0.0002886, 0.000273, 0.0002652, 0.0002574, 0.0002496, 0.000234, 
                                        0.0002262, 0.0002184, 0.000195, 0.0001872, 0.0001638, 0.0001482, 0.0001404, 0.0001326};

            /* convert speed value to duty cycle for pwm signal */
            float conversion(float f_speed); //angle to duty cycle
    }; // class CSpeedingMotor
}; // namespace drivers

#endif// SPEEDINGMOTOR_HPP