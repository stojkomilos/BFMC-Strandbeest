#include <ultrasonic.hpp>

    ultrasonic::ultrasonic(PinName trigPin, PinName echoPin, std::chrono::milliseconds updateSpeed, std::chrono::milliseconds timeout):_trig(trigPin), _echo(echoPin)
    {
        _updateSpeed = updateSpeed;
        _timeout = timeout;
    }

    ultrasonic::ultrasonic(PinName trigPin, PinName echoPin, std::chrono::milliseconds updateSpeed, std::chrono::milliseconds timeout, void onUpdate(int))
    :_trig(trigPin), _echo(echoPin)
    {
        _onUpdateMethod=onUpdate;
        _updateSpeed = updateSpeed;
        _timeout = timeout;
        _t.start ();
    }
    void ultrasonic::_startT()
    {
        if(_t.elapsed_time()>600s)
        {
            _t.reset ();
        }
        start = duration_cast<milliseconds>(_t.elapsed_time());
    }

    void ultrasonic::_updateDist()
    {
        end = duration_cast<milliseconds>(_t.elapsed_time());
        done = 1;
        _distance = (end.count() - start.count())/58; // u centimetrima
        _tout.detach();
        _tout.attach(callback(this, &ultrasonic::_startTrig), _updateSpeed);
    }
    void ultrasonic::_startTrig(void)
    {
            _tout.detach();
            _trig=1;
            wait_us(10);
            done = 0;
            _echo.rise(callback(this, &ultrasonic::_startT));
            _echo.fall(callback(this, &ultrasonic::_updateDist));
            _echo.enable_irq ();
            _tout.attach(callback(this, &ultrasonic::_startTrig),_timeout);
            _trig=0;
    }

    float ultrasonic::getCurrentDistance(void)
    {
        return _distance;
    }

    void ultrasonic::pauseUpdates(void)
    {
        _tout.detach();
        _echo.rise(NULL);
        _echo.fall(NULL);
    }
    void ultrasonic::startUpdates(void)
    {
        _startTrig();
    }
    void ultrasonic::attachOnUpdate(void method(int))
    {
        _onUpdateMethod = method;
    }
    void ultrasonic::changeUpdateSpeed(std::chrono::milliseconds updateSpeed)
    {
        _updateSpeed = updateSpeed;
    }
    std::chrono::milliseconds ultrasonic::getUpdateSpeed()
    {
        return _updateSpeed;
    }
    int ultrasonic::isUpdated(void)
    {
        //printf("%d", done);
        d = done;
        done = 0;
        return d;
    }
    void ultrasonic::checkDistance(void)
    {
        if(isUpdated())
        {
            (*_onUpdateMethod)(_distance);
        }
    }
