import numpy as np
from openbox.utils.config_space import ConfigurationSpace
from openbox.utils.config_space import UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant


class XGBoost:
    def __init__(self, n_estimators, learning_rate, max_depth, min_child_weight,
                 subsample, colsample_bytree, gamma=None, reg_alpha=None, reg_lambda=None,
                 n_jobs=4, seed=47):
        self.n_estimators = int(n_estimators)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.n_jobs = n_jobs
        self.random_state = np.random.RandomState(seed)
        self.estimator = None

    def fit(self, X, y):
        from xgboost import XGBClassifier
        # objective is set automatically in sklearn interface of xgboost
        self.estimator = XGBClassifier(
            use_label_encoder=False,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()
        return self.estimator.predict(X)

    @staticmethod
    def get_cs():
        cs = ConfigurationSpace()
        n_estimators = UniformFloatHyperparameter("n_estimators", 100, 1000, q=50, default_value=500)
        max_depth = UniformIntegerHyperparameter("max_depth", 1, 12)
        learning_rate = UniformFloatHyperparameter("learning_rate", 1e-3, 0.9, log=True, default_value=0.1)
        min_child_weight = UniformFloatHyperparameter("min_child_weight", 0, 10, q=0.1, default_value=1)
        subsample = UniformFloatHyperparameter("subsample", 0.1, 1, q=0.1, default_value=1)
        colsample_bytree = UniformFloatHyperparameter("colsample_bytree", 0.1, 1, q=0.1, default_value=1)
        gamma = UniformFloatHyperparameter("gamma", 0, 10, q=0.1, default_value=0)
        reg_alpha = UniformFloatHyperparameter("reg_alpha", 0, 10, q=0.1, default_value=0)
        reg_lambda = UniformFloatHyperparameter("reg_lambda", 1, 10, q=0.1, default_value=1)
        cs.add_hyperparameters([n_estimators, max_depth, learning_rate, min_child_weight, subsample,
                                colsample_bytree, gamma, reg_alpha, reg_lambda])
        return cs

