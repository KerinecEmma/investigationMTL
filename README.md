# investigationMTL

This code corresponds to the 'When does multi-task learning work for loosely related document classification tasks?' research.
It uses Keras and scikit-learn.

The basic idea is to investigate the performance of single-task and multi-task learning. 
The used models are multi-layer perceptron (with in the case of MTL hard parameter sharing) you can find them in models.py.
The data set used to conduct experiments is 20 Newsgroup, and we proceed classification.
run_models.py is used to conduct tests, the results that I used after that are in the diverse files.
effeciency_prediction.py is used to investigate MTL models and our ability to predict gains when using them (based on different task features).
