from pickle import Pickler

import ray


@ray.remote
class TaskMonitor:
    def __init__(self):
        self.count = 0

    def get_status(self):
        return self.count

    def report(self):
        self.count += 1
        return self.count > 0


@ray.remote
class SimpleCountingDispatcher:
    """
    A common task counter for a worker pool
    """

    def __init__(self, num_tasks):
        self.seqno = 0
        self.num_tasks = num_tasks

    def get_task(self):
        """
        Provide tasks as integers until exhausted
        """
        self.seqno = self.seqno + 1
        if self.seqno <= self.num_tasks:
            return self.seqno
        else:
            return None


@ray.remote
class RayFilePickler:
    def __init__(self, url: str, mode: str):
        self.url = url
        self.file = self._open(mode)


    def _open(self, mode):
        return open(self.url, mode)

    def close(self):
        """
        close the file and commit suicide
        """
        self.file.close()
        ray.actor.exit_actor()

    def write(self, record):
        Pickler(self.file).dump(record)
