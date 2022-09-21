# CNN-Context-Category-Associations
This program takes a CNN model written in PyTorch and runs analyses that reveal its responsiveness to contextual information. It is the code used to produce the results for Aminoff et al. (2022). The code can be run using AlexNet to calculate pearson's correlations and construct a context/category chart from the matrix data from the command line by running:

```
python main.py 
```

There are other flags available for additional analyses. To run all CNN models used in Aminoff et al. (2022), simply run:

```
python main.py -all_models 1
```

If you like this code or find it useful for your work, please cite:

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