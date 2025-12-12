.PHONY: crawl index evaluate app pipeline

PYTHON ?= python3

crawl:
	$(PYTHON) crawler.py

index:
	$(PYTHON) build_index.py

evaluate:
	$(PYTHON) evaluate.py

app:
	streamlit run app.py

pipeline: crawl index evaluate
