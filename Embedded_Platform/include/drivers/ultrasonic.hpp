#ifndef MBED_ULTRASONIC_H
#define MBED_ULTRASONIC_H

#include <mbed.h>
#include <chrono>
using namespace std::chrono;

class ultrasonic
{
    public:
        /**iniates the class with the specified trigger pin, echo pin, update speed and timeout**/
        ultrasonic(PinName trigPin, PinName echoPin, std::chrono::milliseconds updateSpeed, std::chrono::milliseconds timeout);
        /**iniates the class with the specified trigger pin, echo pin, update speed, timeout and method to call when the distance changes**/
        ultrasonic(PinName trigPin, PinName echoPin, std::chrono::milliseconds updateSpeed, std::chrono::milliseconds timeout, void onUpdate(int));
        /** returns the last measured distance**/
        float getCurrentDistance(void);
        /**pauses measuring the distance**/
        void pauseUpdates(void);
        /**starts mesuring the distance**/
        void startUpdates(void);
        /**attachs the method to be called when the distances changes**/
        void attachOnUpdate(void method(int));
        /**changes the speed at which updates are made**/
        void changeUpdateSpeed(std::chrono::milliseconds updateSpeed);
        /**gets whether the distance has been changed since the last call of isUpdated() or checkDistance()**/
        int isUpdated(void);
        /**gets the speed at which updates are made**/
        std::chrono::milliseconds getUpdateSpeed(void);
        /**call this as often as possible in your code, eg. at the end of a while(1) loop,
        and it will check whether the method you have attached needs to be called**/
        void checkDistance(void);
    private:
        DigitalOut _trig;
        InterruptIn _echo;
        Timer _t;
        Timeout _tout;
        float _distance; // in centimeteres
        milliseconds _updateSpeed;
        milliseconds start;
        milliseconds end;
        volatile int done;
        void (*_onUpdateMethod)(int);
        void _startT(void);
        void _updateDist(void);
        void _startTrig(void);
        milliseconds _timeout;
        int d;
};
#endif
