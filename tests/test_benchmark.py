from crosslearner.benchmarks.run_benchmarks import run


def test_run_benchmark_toy_short():
    results = run("toy", replicates=1, epochs=1)
    assert len(results) == 1
    metrics = results[0]
    assert "pehe" in metrics and "policy_risk" in metrics
    assert metrics["pehe"] >= 0.0
