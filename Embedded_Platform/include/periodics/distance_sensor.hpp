/* Include guard */
#ifndef DISTANCE_SENSOR_HPP
#define DISTANCE_SENSOR_HPP

/* The mbed library */
#include <mbed.h>
/* Header file for the task manager library, which  applies periodically the fun function of it's children*/
#include "task.hpp"
#include "ultrasonic.hpp"

namespace periodics
{
    class CDistanceSensor: public utils::CTask
    {
        public:
            CDistanceSensor(
                    uint32_t f_period,
                    PinName,
                    PinName,
                    UnbufferedSerial& f_serial
            );

            ~CDistanceSensor();

        private:
            virtual void        _run();
            ultrasonic *sensor = nullptr;
            UnbufferedSerial&          m_serial;
            bool                m_isActive = false;
    };

}; // namespace periodics

#endif // DISTANCE_SENSOR_HPP
