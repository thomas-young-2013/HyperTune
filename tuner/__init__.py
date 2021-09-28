from tuner.mq_random_search import mqRandomSearch
from tuner.mq_bo import mqBO
from tuner.mq_sh import mqSuccessiveHalving
from tuner.mq_hb import mqHyperband
from tuner.mq_bohb_v0 import mqBOHB_v0
from tuner.mq_bohb_v2 import mqBOHB_v2
from tuner.mq_mfes import mqMFES
from tuner.async_mq_random import async_mqRandomSearch
from tuner.async_mq_bo import async_mqBO
from tuner.async_mq_ea import async_mqEA
from tuner.async_mq_sh import async_mqSuccessiveHalving
from tuner.async_mq_sh_v0 import async_mqSuccessiveHalving_v0
from tuner.async_mq_hb import async_mqHyperband
from tuner.async_mq_hb_v0 import async_mqHyperband_v0
from tuner.async_mq_bohb import async_mqBOHB
from tuner.async_mq_mfes import async_mqMFES

mth_dict = dict(
    # random=(mqRandomSearch, 'sync'),  # sync random (not used)
    bo=(mqBO, 'sync'),  # batch BO
    sh=(mqSuccessiveHalving, 'sync'),  # Successive Halving
    hyperband=(mqHyperband, 'sync'),  # Hyperband
    bohb=(mqBOHB_v0, 'sync'),  # BOHB
    bohbv2=(mqBOHB_v2, 'sync'),  # tpe
    mfeshb=(mqMFES, 'sync'),  # MFES-HB
    arandom=(async_mqRandomSearch, 'async'),  # A-Random
    abo=(async_mqBO, 'async'),  # async batch BO, A-BO
    area=(async_mqEA, 'async', dict(strategy='oldest')),  # Asynchronous Evolutionary Algorithm
    areav2=(async_mqEA, 'async', dict(strategy='worst')),  # Asynchronous Evolutionary Algorithm
    asha=(async_mqSuccessiveHalving_v0, 'async'),  # original asha
    asha_delayed=(async_mqSuccessiveHalving, 'async'),  # delayed asha
    ahyperband=(async_mqHyperband_v0, 'async'),  # A-Hyperband with original asha
    ahyperband_delayed=(async_mqHyperband, 'async'),  # A-Hyperband with delayed asha
    abohb=(async_mqBOHB, 'async'),  # A-BOHB*: our implementation version. prf

    # ours
    tuner=(async_mqMFES, 'async'),

    # exp version
    ahb_bs=(async_mqMFES, 'async', dict(test_random=True,
                                        test_original_asha=True, )),  # A-Hyperband with bracket selection
    abohb_bs=(async_mqMFES, 'async', dict(test_bohb=True,
                                          acq_optimizer='random', )),  # A-BOHB* with bracket selection
    tuner_exp1=(async_mqMFES, 'async', dict(use_weight_init=False, )),  # test ours without bracket selection
    tuner_exp2=(async_mqMFES, 'async', dict(test_original_asha=True, )),  # test original asha + ours
)
