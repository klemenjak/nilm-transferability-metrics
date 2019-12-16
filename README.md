# Transferability Metrics for NILM

This repository is going to contain the implementation of our transferability metrics, which we define in our latest publication: [On Metrics to Assess the Transferability of Machine
Learning Models in Non-Intrusive Load Monitoring](https://arxiv.org/pdf/1912.06200.pdf)

PS: compatible to NILMTK!

```python
from GenLoss import *
from nilmtk.api import API
from file_handler import load_experiment

from nilmtk.disaggregate import FHMMExact, Hart85
from nilmtk_contrib import *

# load dict and execute experiment
experiment = load_experiment(experiment_ID)
api_results = API(experiment)

# assess generalisation abilities
g_loss = mean_generalization_loss(api_results)
auh = accuracy_on_unseen_houses(api_results)

exit()
```
