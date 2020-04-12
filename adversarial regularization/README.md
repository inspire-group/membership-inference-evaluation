### About
This is the defense method proposed in the [*ACM CCS* 2018 paper](https://arxiv.org/abs/1807.05852).  
If needed, please follow [their released code](https://github.com/SPIN-UMass/ML-Privacy-Regulization) to train both natural and defended models.  
Alternatively, you can also download our pretrained models.

### Pretrained Models  
Purchase100 Dataset: [natural model](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/purchase_natural); 
[defended model](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/purchase_advreg); 
[model with early stopping](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/purchase_early_stop)  
Texas100 Dataset: [natural model](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/texas_natural); 
[defended model](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/texas_advreg); 
[model with early stopping](http://www.princeton.edu/~liweis/membership-inference-evaluation/AdvReg/texas_early_stop)  

### Usage
`python MIA_evaluate.py --dataset` [*purchase* or *texas*] `--model-dir` [*the path of target classifier*]
