# RAG API with vLLM - Makefile

SHELL := /bin/bash
PYTHON := python
VENV := source vllm-env/bin/activate &&
BENCHMARK_DIR := benchmarks

.PHONY: help install vllm api health query test benchmark visualize clean

help:
	@echo "RAG API with vLLM - Commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make vllm       - Start vLLM server"
	@echo "  make api        - Start RAG API server"
	@echo "  make health     - Health check"
	@echo "  make query      - Test query"
	@echo "  make test       - Quick test"
	@echo "  make benchmark  - GPU benchmark"
	@echo "  make visualize  - Visualize results"
	@echo "  make clean      - Clean generated files"

install:
	@python3 -m venv vllm-env 2>/dev/null || true
	@$(VENV) pip install --upgrade pip setuptools wheel
	@$(VENV) pip install -r requirements.txt
	@mkdir -p data benchmarks/results benchmarks/reports
	@echo "✓ Installation complete"

vllm:
	@$(VENV) vllm serve Qwen/Qwen2.5-3B-Instruct --port 8082 --gpu-memory-utilization 0.85

api:
	@$(VENV) $(PYTHON) main.py

health:
	@curl -s http://localhost:8000/health | python -m json.tool || echo "API not running"
	@curl -s http://localhost:8082/health | python -m json.tool || echo "vLLM not running"

query:
	@curl -s -X POST http://localhost:8000/query \
		-H "Content-Type: application/json" \
		-d '{"question": "Yazilim muhendisligi nedir?"}' | python -m json.tool

test:
	@$(VENV) cd $(BENCHMARK_DIR) && $(PYTHON) concurrent_test.py

benchmark:
	@$(VENV) cd $(BENCHMARK_DIR) && $(PYTHON) benchmark.py --test-type load

visualize:
	@$(VENV) cd $(BENCHMARK_DIR) && $(PYTHON) visualize_results.py

clean:
	@rm -rf $(BENCHMARK_DIR)/results/*.json $(BENCHMARK_DIR)/reports/*.png
	@rm -rf __pycache__ $(BENCHMARK_DIR)/__pycache__
	@rm -f rag_api.log
	@echo "✓ Cleaned"
