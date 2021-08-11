# Multistream Graph Attention Networks

Implementation code for the paper Multistream Graph Attention Networks.
The experimental setup and data is forked from [WeatherGCNet](https://github.com/tstanczyk95/WeatherGCNet).

![gat](https://user-images.githubusercontent.com/38927466/129062241-cd8b1739-0fdc-49c4-9420-e48f8cbbd1b4.jpg)

## Training ##

```
python train_model.py -dp data/dataset.pkl -e 50 -it 30 -pt 2
```

## Testing ##

```
python test_model.py -dp data/dataset.pkl -it 30 -pt 2 -mp trained_models/2h_pred_model.pt
```
