# Multistream Graph Attention Networks

Implementation code for the paper Multistream Graph Attention Networks.
The experimental setup and data is forked from [WeatherGCNet](https://github.com/tstanczyk95/WeatherGCNet).

![gat](https://user-images.githubusercontent.com/38927466/129062241-cd8b1739-0fdc-49c4-9420-e48f8cbbd1b4.jpg)

## Training ##

### Dutch Cities ###
```
python train_model.py -dp data/dataset.pkl -e 50 -it 30 -pt 2
```

### Danish Cities ###
```
python train_model.py -dp data/step1.mat -e 50 
```

## Testing ##

### Dutch Cities ###
```
python test_model.py -dp data/dataset.pkl -it 30 -pt 2 -mp trained_models/2h_pred_model.pt
```

### Danish Cities ###
```
python test_model.py -dp data/step1.mat -mp trained_models/6h_pred_model.pt
```
