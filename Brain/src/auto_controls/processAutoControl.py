if __name__ == "__main__":
    import sys
    sys.path.insert(0, "../..")

from src.templates.workerprocess import WorkerProcess
from src.auto_controls.threads.threadAutoControl import threadAutoControl
from multiprocessing import Pipe

import sys
sys.path.append("src/auto_controls/threads")


class processAutoControl(WorkerProcess):
    """This process controls the car
    Args:
        queueList (dictionar of multiprocessing.queues.Queue): Dictionar of queues where the ID is the type of messages.
        logger (logging object): Made for debugging.
        debugging (bool, optional): A flag for debugging. Defaults to False.
    """

    def __init__(self, queueList, logger, debugging=False):
        self.logger = logger
        self.debugging = debugging
        self.queueList = queueList

        pipeRecv, pipeSend = Pipe(duplex=False)
        self.pipeRecv = pipeRecv
        self.pipeSend = pipeSend

        pipeEngineRunRecv, pipeEngineRunSend = Pipe(duplex=False)
        self.pipeEngineRunRecv = pipeEngineRunRecv
        self.pipeEngineRunSend = pipeEngineRunSend

        self.pipeReceiveCars, self.pipeSendCars = Pipe(duplex=False)

        self.pipeReceiveSemaphores, self.pipeSendSemaphores = Pipe(duplex=False)

        super(processAutoControl, self).__init__(self.queueList)

    # ===================================== STOP ==========================================

    def stop(self):
        """Function for stopping threads and the process."""
        for thread in self.threads:
            thread.stop()
            thread.join()
        super(processAutoControl, self).stop()

    # ===================================== RUN ===========================================
    def run(self):
        """Apply the initializing methods and start the threads."""
        super(processAutoControl, self).run()

    # ===================================== INIT TH ==========================================
    def _init_threads(self):
        """Initializes the gateway thread."""
        gatewayThread = threadAutoControl(self.queueList, self.logger, self.debugging, self.pipeRecv, self.pipeSend, self.pipeEngineRunRecv, self.pipeEngineRunSend, self.pipeReceiveCars, self.pipeSendCars, self.pipeReceiveSemaphores, self.pipeSendSemaphores)
        self.threads.append(gatewayThread)
