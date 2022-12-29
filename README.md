# ironloss_prediction_of_IPMSM_by_XGBoost

Python implementation for iron loss prediction of interior permanent magnet synchronous motors by XGBoost ([paper](https://ieeexplore.ieee.org/document/10002362))

## Overview
This library contains an implementation of an automatic design system for interior permanent magnet synchronous motors as presented in [1]

## Dependencies
- python>=3.8
- numpy
- pandas
- sklearn>=0.24.2
- xgboost>=1.6.1
- optuna>=2.10.1

## Architecture
data: You can download the dataset used for the paper [here](https://ieee-dataport.org/documents/dataset-iron-losses-ipmsms).  
[regression](/regression.py): A characteristics prediction model is implemented.


## Feedback
For questions and comments, feel free to contact [Yuki Shimizu](yshimizu@fc.ritsumei.ac.jp).

## License
MIT

## Citation
```
[1] Y. Shimizu, “Efficiency Optimization Design that Considers Control 
of Interior Permanent Magnet Synchronous Motors based on Machine Learning 
for Automotive Application,” IEEE Access, Accepted
```
