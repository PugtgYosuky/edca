# EDCA - Evolutionary Data Centric AutoML

## Authors

- Joana Simões, <joanasimoes@student.dei.uc.pt>
- João Correia, <jncor@dei.uc.pt>
  
## What is EDCA?

EDCA is a Python library for Automated Machine Learning (AutoML). It optimizes the entire Machine Learning (ML) pipeline. Given a classification dataset, EDCA starts by making an analysis of the features types and characteristics. This analysis serves to define the data transformations required for the data in question.
Then, with the pipeline steps required, it starts the search for its bests estimators and models for each step of the pipeline. The search relies on a Genetic Algorithm. In the end, the user receives the best pipeline found trained ready to make prediction about unseen data.

## Installation

1. Clone or download the EDCA's GitHub Repository
2. Install EDCA from the main repository directory with

        pip install -e .

Note: the *tests* directory contains notebook with use case examples.

## Acknowledgements

EDCA was developed in the context of a Master Dissertation in Engineering and Data Science at the Department of Informatics and Engineering in University of Coimbra, Portugal.
