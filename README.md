### Contextual associations represented in Convolutional Neural Networks
This program takes a PyTorch CNN image classification model and runs analyses that reveal its responsiveness to contextual information at each layer. It is the code used to produce the results for [Aminoff et al. (2022)](https://www.nature.com/articles/s41598-022-09451-y). The code can be run using Vgg16 to calculate pearson's correlations and construct a context/category chart from the matrix data from the command line by running:

```
python main.py -vgg16 1
```

Similarity ratios for the neural network analysis are then calculated using the following formula: 
<p align='center'>
  <img src="https://latex.codecogs.com/svg.image?SimRatio^C&space;=&space;\frac{MeanInSim^C}{MeanOutSim^C}&space;=&space;\frac{\frac{1}{N_{inGroup}^C}\sum_{(i,j)\in&space;C,&space;i&space;\neq&space;j}sim(p_{i},&space;p_{j})}{\frac{1}{N_{outGroup}^C}\sum_{i\in&space;C,j\in&space;C^{\prime}}sim(p_{i},&space;p_{j})}" title="https://latex.codecogs.com/svg.image?SimRatio^C = \frac{MeanInSim^C}{MeanOutSim^C} = \frac{\frac{1}{N_{inGroup}^C}\sum_{(i,j)\in C, i \neq j}sim(p_{i}, p_{j})}{\frac{1}{N_{outGroup}^C}\sum_{i\in C,j\in C^{\prime}}sim(p_{i}, p_{j})}" />
</p>

There are other flags available for additional analyses. To run all CNN models used in Aminoff et al. (2022), simply run:

```
python main.py -all_models 1
```

To assess visual similarity of categories within the data, run:
```
python main.py -hog_pixel_similarity 1
```

### Cite
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
