### Contextual associations represented in Convolutional Neural Networks
This program takes a PyTorch CNN image classification model and runs analyses that reveal its relationships between objects-scenes at each layer.

```
python main.py -alexnet 1
```

Similarity ratios for the neural network analysis are then calculated using the following formula: 
<p align='center'>
  <img src="https://latex.codecogs.com/svg.image?SimRatio^C&space;=&space;\frac{MeanInSim^C}{MeanOutSim^C}&space;=&space;\frac{\frac{1}{N_{inGroup}^C}\sum_{(i,j)\in&space;C,&space;i&space;\neq&space;j}sim(p_{i},&space;p_{j})}{\frac{1}{N_{outGroup}^C}\sum_{i\in&space;C,j\in&space;C^{\prime}}sim(p_{i},&space;p_{j})}" title="https://latex.codecogs.com/svg.image?SimRatio^C = \frac{MeanInSim^C}{MeanOutSim^C} = \frac{\frac{1}{N_{inGroup}^C}\sum_{(i,j)\in C, i \neq j}sim(p_{i}, p_{j})}{\frac{1}{N_{outGroup}^C}\sum_{i\in C,j\in C^{\prime}}sim(p_{i}, p_{j})}" />
</p>

There are other flags available for additional analyses. To run all CNN models used in Aminoff et al. (2022), simply run:

```
python main.py -a 1
```


### Cite
If you find this code useful for your work, please cite:

```
