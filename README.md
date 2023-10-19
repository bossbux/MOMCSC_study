# Multi-Objective Optimization Study (MOMCSC_study.py)

## Table of Contents
- [Description](#description)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Additional Notes](#additional-notes)

## Description
This Python script, `MOMCSC_study.py`, is part of a study that evaluates various multi-objective optimization algorithms. It tests the performance of the following algorithms:
- NSGAII
- pRTMNSGAII
- RNSGAIII
- TMNSGAIII
- NSGAIII
- pNSGAII
- NSPSO

The study is designed to address the multi-objective multi-cloud service selection problem. It provides the flexibility to work with datasets of varying numbers of clouds and services.

## Prerequisites
- Python 3.10.7

## Usage
1. Set up your Python environment with Python 3.10.7.

2. Place the dataset you want to use for the study in the specified location (line 469 in the script) and provide the absolute path to the dataset.

3. Ensure you have an empty folder where the study results will be stored. This folder should be created in advance and specified in the script.

4. Run the `MOMCSC_study.py` script.

5. The study will use pre-implemented algorithms from the `pymoo` library, with additional algorithms implemented in the `pymoo` infrastructure. The modified `pymoo` library is included in the attached files.

## Folder Structure
- `pymoo` (Modified `pymoo` library with additional algorithms)
- `MOMCSC_study.py` (Main Python script for the study)
- `dataset` (Location for the dataset)
- `results` (Empty folder for study results)

## Additional Notes
The main modifications from the original `pymoo` library are found in the following files:
- `pymoo/factory.py`
- `pymoo/algorithms/moo`

