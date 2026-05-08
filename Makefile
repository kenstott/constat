.PHONY: test test-parallel test-sequential

# Run all non-sequential tests in parallel (default)
test-parallel:
	pytest -m "not run_sequentially" -n auto

# Run sequential (embedding/reranking) tests one at a time
test-sequential:
	pytest -m "run_sequentially" -p no:xdist

# Run full suite: parallel first, then sequential
test: test-parallel test-sequential
