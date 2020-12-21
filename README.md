### About
This code accompanies the paper "Systematic Evaluation of Privacy Risks of Machine Learning Models", accepted by USENIX Security 2021.

### Usage
`membership_inference_attacks.py` contains the main membership inference attack code;  
`privacy_risk_score_utils.py` contains the code to compute the privacy risk score for each individual sample.

In each folder, `MIA_evaluate.py` performs attacks against target machine learning classifiers.  
If you want to further compute the privacy risk score, first import `privacy_risk_score_utils.py`; after initializing the attack class in `MIA_evaluate.py`, add `risk_score = calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels, MIA.s_te_labels, MIA.t_tr_m_entr, MIA.t_tr_labels)`

### Impact 
Our evaluation methods have been intergrated into [Google's TensorFlow Privacy library](https://github.com/tensorflow/privacy), including both [attack methods](https://github.com/tensorflow/privacy/pull/131) and the [fine-grained individual privacy risk analysis](https://github.com/tensorflow/privacy/pull/146).
