import abc
import logging

import ray

from pickle import Pickler


logger = logging.getLogger(__name__)


LOG_LEVEL = logging.WARNING


class StatefulRayPolicy:
    """
    Common Superclass of deployable policies. These policies are built such that they contruct
    non-serializable elements only after having been deployed
    """
    @abc.abstractmethod
    def init(self, **init_args):
        pass

    @abc.abstractmethod
    def get_winner(self, state):
        pass

    @abc.abstractmethod
    def get_advisable_actions(self, state):
        pass

    @abc.abstractmethod
    def predict(self, state):
        pass

    @abc.abstractmethod
    def load_checkpoint(self, folder, filename, **kwargs):
        pass


@ray.remote
class PolicyWorker:
    def __init__(self, policy, **init_args):
        logging.basicConfig(level=LOG_LEVEL)
        self.policy = policy
        self.initialized = False
        self.init_args = init_args
        self.logger = logging.getLogger(__name__)
        self.logger.info("Up and running.")

    def init(self):
        self.policy.init(**self.init_args)

    def get_winner(self, state):
        return self.policy.get_winner(state)

    def get_advisable_actions(self, state):
        return self.policy.get_advisable_actions(state)

    def predict(self, state):
        return self.policy.predict(state)

    def load_checkpoint(self, folder, filename):
        return self.policy.load_checkpoint(folder, filename)


@ray.remote
class PolicyDispatcher:
    """
    Round-Robin Dispatcher
    """
    def __init__(self, workers):
        logging.basicConfig(level=LOG_LEVEL)
        self.workers = workers
        self.logger = logging.getLogger(__name__)
        self.i = 0
        self.m = len(workers)

    def next_actor(self):
        self.i = (self.i + 1) % self.m
        return self.workers[self.i]

    def predict(self, state):
        w = self.next_actor()
        return w.predict.remote(state)

    def get_advisable_actions(self, state):
        w = self.next_actor()
        return w.get_advisable_actions.remote(state)

    def get_winner(self, state):
        w = self.next_actor()
        return w.get_winner.remote(state)


@ray.remote
class SelfPlayDelegator:
    def __init__(self, wid, file_writer, counter, delegate):
        self.wid = wid
        self.file_writer = file_writer
        self.delegate = delegate
        self.counter = counter

    def work(self):
        """
        fetch task, report result - until all jobs are done
        """
        while True:
            seqno = ray.get(self.counter.get_task.remote())
            if seqno is None:
                break
            the_result = ray.get(self.delegate.observe_trajectory.remote(for_storage=True))
            self.file_writer.write.remote(the_result)


class PolicyRef:
    """
    Regular blocking client. Calls the actor and unwraps the result
    """
    def __init__(self, actor):
        self.actor = actor

    def predict(self, x):
        y = self.actor.predict.remote(x)
        while isinstance(y, ray._raylet.ObjectRef):  # noqa
            y = ray.get(y)
        return y

    def get_advisable_actions(self, x):
        y = self.actor.get_advisable_actions.remote(x)
        while isinstance(y, ray._raylet.ObjectRef):  # noqa
            y = ray.get(y)
        return y

    def get_winner(self, x):
        y = self.actor.get_winner.remote(x)
        while isinstance(y, ray._raylet.ObjectRef):  # noqa
            y = ray.get(y)
        return y


def create_pool(num_workers, policy: StatefulRayPolicy, **init_args):
    """
    Create a pool of policy actors
    :param num_workers:
    :param policy:
    :param init_args: Arguments to pass to each worker
    :return: A reference to its _dispatcher actor
    """
    logging.basicConfig(level=LOG_LEVEL)
    workers = []
    for i in range(num_workers):
        w = PolicyWorker.remote(policy=policy, **init_args)

        # wait for the the initialization to succeed
        ray.get(w.init.remote())  # noqa
        workers.append(w)

    d = PolicyDispatcher.remote(workers=workers)
    logger.info(f"Dispatcher launched successfully")
    return d
