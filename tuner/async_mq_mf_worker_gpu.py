from functools import partial
from tuner.async_mq_mf_worker import async_mqmfWorker


class async_mqmfWorker_gpu(async_mqmfWorker):
    """
    async message queue worker for multi-fidelity optimization
    gpu version: specify 'device'
    """
    def __init__(self, objective_function,
                 device,
                 ip="127.0.0.1", port=13579, authkey=b'abc',
                 sleep_time=0.1,
                 no_time_limit=False,
                 logger=None):
        objective_function = partial(objective_function, device=device)
        super().__init__(
            objective_function=objective_function,
            ip=ip, port=port, authkey=authkey,
            sleep_time=sleep_time,
            no_time_limit=no_time_limit,
            logger=logger
        )
        self.logging('Worker device: %s' % device)
