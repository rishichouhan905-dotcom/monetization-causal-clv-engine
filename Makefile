PYTHON_VERSION := 3.13.14
PYTHON := python

.PHONY: check-env setup data dbt clv causal report app test all results demo sample montecarlo

check-env:
	@$(PYTHON) -c "import sys; assert sys.version_info[:2] == (3, 13), f'Expected Python 3.13.*, found {sys.version}'"

setup: check-env
	pip install -e .

data: check-env
	$(PYTHON) scripts/download_data.py

dbt: check-env
	$(PYTHON) src/clv_causal/warehouse/build_warehouse.py

clv: check-env
	$(PYTHON) src/clv_causal/clv_engine/clv_pipeline.py

causal: check-env
	$(PYTHON) src/causal/evaluation.py

results: check-env
	@$(PYTHON) -c "import os; os.makedirs('results', exist_ok=True)"

report: check-env results
	$(PYTHON) scripts/build_clv_report.py

montecarlo: check-env results
	$(PYTHON) src/causal/monte_carlo.py

app: check-env
	$(PYTHON) -m streamlit run src/clv_causal/app/main.py

demo: check-env
	$(PYTHON) -m streamlit run src/clv_causal/app/main.py

sample: check-env
	$(PYTHON) scripts/create_sample.py

test: check-env
	pytest

all: data dbt clv causal report test
