# Rafiki Lifelong Learning

``` bash
docker run -it -u root -v $(pwd):/app/codalab codalab/codalab-legacy:py3 bash

# For ingestion
python3 AutoML3_ingestion_program/ingestion.py AutoML3_sample_data AutoML3_sample_predictions AutoML3_sample_ref AutoML3_ingestion_program rafiki_lifelong

# For scoring
python3 AutoML3_scoring_program/score.py 'AutoML3_sample_data/*/' AutoML3_sample_predictions AutoML3_scoring_output
```
