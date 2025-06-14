from crosslearner.benchmarks.run_benchmarks import run


def test_run_benchmark_toy_short():
    results = run("toy", replicates=1, epochs=1)
    assert len(results) == 1
    assert results[0] >= 0.0
