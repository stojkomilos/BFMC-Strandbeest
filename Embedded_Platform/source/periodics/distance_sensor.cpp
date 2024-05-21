
#include "distance_sensor.hpp"
#include <chrono>
#include <mbed.h>

namespace periodics
{
    CDistanceSensor::CDistanceSensor(
        uint32_t f_period,
        PinName echo,
        PinName trig,
        UnbufferedSerial& f_serial)
        : utils::CTask(f_period), m_serial(f_serial), m_isActive(true) {
        this->sensor = new ultrasonic(trig, echo, std::chrono::milliseconds(f_period), std::chrono::milliseconds(32));
    }

    CDistanceSensor::~CDistanceSensor() {}

    void CDistanceSensor::_run() {

        sensor->checkDistance();
        auto diff = sensor->getCurrentDistance();
        char buffer[256];
        float l_rps = diff;
        snprintf(buffer, sizeof(buffer), "@12:%.1f;;\r\n", l_rps);
        m_serial.write(buffer,strlen(buffer));

    }

}; // namespace periodics
