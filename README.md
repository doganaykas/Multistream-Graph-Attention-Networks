# Multistream Graph Attention Networks

Implementation code for the paper Multistream Graph Attention Networks.
The experimental setup and data is forked from [WeatherGCNet](https://github.com/tstanczyk95/WeatherGCNet).

![gat](https://user-images.githubusercontent.com/38927466/129062241-cd8b1739-0fdc-49c4-9420-e48f8cbbd1b4.jpg)

## 🖥️ Training ##

### Dutch Cities ###
```
python train_model.py -dp data/dataset.pkl -e 50 -it 30 -pt 2
```

### Danish Cities ###
```
python train_model.py -dp data/step1.mat -e 50 
```

## 🔍 Testing ##

### Dutch Cities ###
```
python test_model.py -dp data/dataset.pkl -it 30 -pt 2 -mp trained_models/2h_pred_model.pt
```

### Danish Cities ###
```
python test_model.py -dp data/step1.mat -mp trained_models/6h_pred_model.pt
```

## 📊 Results ##

![g1](https://user-images.githubusercontent.com/38927466/129238554-d9d2f281-33ca-4fae-9bfc-d1dc2daa64c0.JPG)
![g3](https://user-images.githubusercontent.com/38927466/129238564-38c770ae-efe2-443c-8a7b-7674fce3bf33.JPG)


![nl2](https://user-images.githubusercontent.com/38927466/129239241-03e80bf5-1d2b-49c9-8399-c44704a5f773.JPG)
![den4](https://user-images.githubusercontent.com/38927466/129239245-cba8b82f-8422-4dab-9f58-eec0558c841e.jpg)

## 🔗 Citation ##
For citation, please use the bibtex reference below:

```
@article{MGAT2021,
  title={Multistream Graph Attention Networksfor Wind Speed Forecasting},
  author={Aykas, Dogan and Mehrkanoon, Siamak},
  journal={arXiv preprint arXiv:2108.07063}
  year={2021}
}
```
