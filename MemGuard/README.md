### About
This is the defense method proposed in the [*ACM CCS* 2019 paper](https://arxiv.org/abs/1909.10594)  
If needed, please follow [their release code](https://github.com/jjy1994/MemGuard) to obtain model predictions with (or without) adversarial noises. 
(Note the source code only runs defense on the target data, you should use the same method to also apply defense on the shadow data.)  
Alternatively, we also provide the prediction results in the folder of "saved_predictions".

### Usage
`python MIA_evaluate.py --dataset` [*location* or *texas*] `--defended` [*1* or *0*]

### Others
If you test Texas dataset, please first download the data [here](https://drive.google.com/file/d/1XAyBj2DJB6BHvyHJhYPUrUkYsORamclq/view?usp=sharing) 
and put it into "data/texas/"
