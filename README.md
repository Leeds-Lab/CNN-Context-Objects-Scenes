# Contextual associations represented in Convolutional Neural Networks
This program takes a PyTorch CNN image classification model and runs analyses that reveal its responsiveness to contextual information at each layer. It is the code used to produce the results for [Aminoff et al. (2022)](https://www.nature.com/articles/s41598-022-09451-y). The code can be run using AlexNet to calculate pearson's correlations and construct a context/category chart from the matrix data from the command line by running:

```
python main.py -alexnet 1
```

There are other flags available for additional analyses. To run all CNN models used in Aminoff et al. (2022), simply run:

```
python main.py -all_models 1
```

If you find this code useful for your work, please cite:

```
@article{aminoff2022contextual,
  title={Contextual associations represented both in neural networks and human behavior},
  author={Aminoff, Elissa M and Baror, Shira and Roginek, Eric W and Leeds, Daniel D},
  journal={Scientific reports},
  volume={12},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```
