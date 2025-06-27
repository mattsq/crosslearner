import optuna
from crosslearner.sweep import _space


class DummyTrial(optuna.trial.Trial):
    def __init__(self):
        self._trial_id = 0

    def _suggest(self, name, distribution):
        if name == "disentangle":
            return True
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            return distribution.low
        if isinstance(distribution, optuna.distributions.IntDistribution):
            return distribution.low
        return distribution.choices[0]

    def suggest_float(self, name, low, high=None, *, step=None, log=False):
        dist = optuna.distributions.FloatDistribution(low, high, step=step, log=log)
        return self._suggest(name, dist)

    def suggest_int(self, name, low, high=None, *, step=1, log=False):
        dist = optuna.distributions.IntDistribution(low, high, step=step, log=log)
        return self._suggest(name, dist)

    def suggest_categorical(self, name, choices):
        dist = optuna.distributions.CategoricalDistribution(choices)
        return self._suggest(name, dist)


def test_space_disentangle_includes_rep_dims():
    params = _space(DummyTrial())
    assert params["disentangle"]
    assert all(k in params for k in ("rep_dim_c", "rep_dim_a", "rep_dim_i"))
