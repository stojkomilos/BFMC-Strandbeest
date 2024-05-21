import threading
import base64

from src.templates.threadwithstop import ThreadWithStop
from src.utils.messages.allMessages import *
from multiprocessing import Pipe

from src.auto_controls.threads.runner import Runner

import globals

class threadAutoControl(ThreadWithStop):
    """

    This threadonly gathers data from sesnsors and camera and sends it to be processed by other functions.

    def subscribe      -> subcribes to camera
    def get_camera_msg -> checks if camera send a picture and gets it
    def send           -> send speed and angle to ThreadWrite and move vehicle

    
    queueList          -> message channels
    logger             -> logging object
    
    
    """

    # ===================================== INIT =========================================
    def __init__(self, queueList, logger, debugging, prcv, psnd, pipeEngineRunRecv, pipeEngineRunSend, pipeReceiveCars, pipeSendCars, pipeReceiveSemaphores, pipeSendSemaphores):
        super(threadAutoControl, self).__init__()

        self.queueList = queueList
        self.logger = logger

        # Pipe for camera
        self.pipeReceiveSerialCamera = prcv
        self.pipeSendSerialCamera = psnd

        # Pipe for engine run
        self.pipeEngineRunSend = pipeEngineRunSend
        self.pipeEngineRunRecv = pipeEngineRunRecv

        # Pipe for cars data 
        self.pipeSendCars = pipeSendCars
        self.pipeReceiveCars = pipeReceiveCars

        # Pipe for semaphores data 
        self.pipeSendSemaphores = pipeSendSemaphores
        self.pipeReceiveSemaphores = pipeReceiveSemaphores

        # Obj for lane detection
        # self.lane_detection = LaneDetection()

        # print("Jedan")

        self.runner = Runner()

        self.subscribe()

    def subscribe(self):
        """
        SUBSCRIBE TO CAMERA AND SENSORS
        """
        self.queueList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": serialCamera.Owner.value,
                "msgID": serialCamera.msgID.value,
                "To": {"receiver": "threadAutoControl", "pipe": self.pipeSendSerialCamera},
            }
        )
        """
        SUBSCRIBE TO cars 
        """
        self.queueList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Cars.Owner.value,
                "msgID": Cars.msgID.value,
                "To": {"receiver": "threadAutoControl", "pipe": self.pipeSendCars},
            }
        )
        """
        SUBSCRIBE TO semaphores 
        """
        self.queueList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": Semaphores.Owner.value,
                "msgID": Semaphores.msgID.value,
                "To": {"receiver": "threadAutoControl", "pipe": self.pipeSendSemaphores},
            }
        )
        """
        subscribe to engine_run notif
        """
        self.queueList["Config"].put(
            {
                "Subscribe/Unsubscribe": "subscribe",
                "Owner": EngineRun.Owner.value,
                "msgID": EngineRun.msgID.value,
                "To": {"receiver": "threadAutoControl",
                        "pipe": self.pipeEngineRunSend},
            }

        )


    def get_camera_msg(self):

        if not self.pipeReceiveSerialCamera.poll():
            return False, None, None, None, None

        msg = self.pipeReceiveSerialCamera.recv()
        """
        Message format:
        {
            "value" : image_data_encoded = base64.b64encode(encoded_img).decode("utf-8")
            "Type"  : ()
            "Owner" : ()
            "id"    : ()
        }
        """

        return True, msg["value"], msg["Type"], msg["Owner"], msg["id"]

    def get_cars_msg(self):

        if not self.pipeReceiveCars.poll():
            return False, None, None, None

        msg = self.pipeReceiveCars.recv()
        """
        Message format:
        {
            "value" : {"id" : id, "x" : x, "y" : y}
            "Type"  : ()
            "Owner" : ()
            "id"    : ()
        }
        """
        msg = msg["value"]

        print(msg)
        return True, msg["id"], msg["x"], msg["y"]
 
    def get_semaphores_msg(self):

        if not self.pipeReceiveSemaphores.poll():
            return False, None, None, None, None

        msg = self.pipeReceiveSemaphores.recv()
        """
        Message format:
        {
            "value" : {"id" : id, "x" : x, "y" : y, "state" : state}
            "Type"  : ()
            "Owner" : ()
            "id"    : ()
        }
        """
        msg = msg["value"] 
        print(msg)

        return True, msg["id"], msg["x"], msg["y"], msg["state"]   

    def get_engine_run_msg(self):
        if not self.pipeEngineRunRecv.poll():
            return None
        msg = self.pipeEngineRunRecv.recv()
        """
        Message format:
        {
            "value" : bool
            "Type"  : ()
            "Owner" : ()
            "id"    : ()
        }
        """
        return msg["value"]

    

    def send(self, angle: float, speed: float, image):
        self.queueList[AutoSpeedMotor.Queue.value].put(
            {
                    "Owner": AutoSpeedMotor.Owner.value,
                    "msgID": AutoSpeedMotor.msgID.value,
                    "msgType": AutoSpeedMotor.msgType.value,
                    "msgValue": speed
            }
        )

        self.queueList[AutoSteerMotor.Queue.value].put(
            {
                    "Owner": AutoSteerMotor.Owner.value,
                    "msgID": AutoSteerMotor.msgID.value,
                    "msgType": AutoSteerMotor.msgType.value,
                    "msgValue": angle
            }
        )

        # this part of code sends image
        if image is not None: # if image is None, than that means that we are not displaying the debug stuff
            self.queueList[LineDetectionCamera.Queue.value].put(
                {
                    "Owner": LineDetectionCamera.Owner.value,
                    "msgID": LineDetectionCamera.msgID.value,
                    "msgType": LineDetectionCamera.msgType.value,
                    "msgValue": image,
                }
            )

    def run(self):
        # pass
        while self._running:
            self.runner.do_iteration_online(self)
            # sem = self.get_semaphores_msg()
            # if :
            #     print(sem)
            # cars = self.get_cars_msg()
            # if got_msg:
            #     print(cars)