# HeartRate

This a heart rate estimation using data from PPG and acceleration data using from study:
Z. Zhang, Z. Pi, B. Liu, TROIKA: A general framework for heart rate monitoring using wrist-type photoplethysmographic signals during intensive physical exercise, IEEE Transactions on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015, DOI: 10.1109/TBME.2014.2359372

Data is available here:
https://zenodo.org/records/3902710

Estimation using 1-D convolutional nnet:
ppg_ieee.ipynb

Same with bandpass filtering of the input data:
ppg_ieee_bp.ipynb

Same with windowing data:
ppg_ieee_windowed.ipynb

An attempt to use transformers for improvements, which didn't work well. May be nnet configuration problem or too little data, or both.
ppg_ieee_transf.ipynb
