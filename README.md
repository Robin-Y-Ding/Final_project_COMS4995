# Defensive Perturbation to Improve Robustness of Deep Neural Networks.
### Code for final project of COMS4995 Deep Learning for Computer Vision.
### Usage
#### Results for RQ1:
1. Train the model on MNIST: `python3 standard_training.py`. The outputs will contain the accuracy on non-training data without any attack.
2. Attack the model: 1) Linf-PGD: `python3 dp.py -p linf -k 20 -e 0.2 -a 0.01` 2) L2-PGD: `python3 dp.py -p linf -k 40 -e 2.5 -a 0.075`. The output will fistly show the attack results on original data, and then generate defensive perturbation and evaluate the accuracy on defensive data.
#### Results for RQ2:
Please call the method `draw_tsne(inputs, labels)` defined at `line 216-line 224` in `utils.py` when you hope to plot data points of any version of data using tSNE. A demo example to teach users how to draw tSNE plots is provided at line `152-155` in `standard_training.py`. Please reverse the comments and run the code to see the demo. Any numpy data with labels can be plotted by this method.

Due to the reproduction is very complicated and time-consuming, I didn't automate this process in the current version. Please check the video and the final report for the full results. Sorry for any incovenience.
