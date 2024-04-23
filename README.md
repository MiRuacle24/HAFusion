# HAFusion: Urban Region Representation Learning with Attentive Fusion (ICDE 2024)

This is a pytorh implementation of the [HAFusion paper](https://arxiv.org/abs/2312.04606)

Authors: Fengze Sun, Jianzhong Qi, Yanchuan Chang, Xiaoliang Fan, Shanika Karunasekera, and Egemen Tanin

## Model Structure
<p align="center">
    <img src="Images/model structure.png" width="700">
</p>

## Experiments
<p align="center"><strong>Overall Prediction Accuracy Results</strong></p>
<p align="center">
    <img src="Images/Experiment.png" width="700"> 
</p>

<p align="center"><strong>Prediction Accuracy Results When Powering Existing Models with Our DAFusion Module (NYC)</strong></p>
<div align="center">
    <img src="Images/DAFusion.png" width="700"> 
</div>

## Requirements
- Python 3.8.18
- `pip install -r requirements.txt`

## Quick Start
To train and test HAFusion on a specific city and a specific downstream task:

- CITY_NAME: <strong>NY</strong> or <strong>Chi</strong> or <strong>SF</strong>
- TASK_NAME: <strong>checkIn</strong> or <strong>crime</strong> or <strong>serviceCall</strong>

```bash
python HAFusion_train.py --city CITY_NAME --task TASK_NAME
```

## Contact
Email fengzes@student.unimelb.edu.au if you have any queries.
