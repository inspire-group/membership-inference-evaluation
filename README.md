### About
Systematic Evaluation of Membership Inference Privacy Risks of Machine Learning Models.

### Usage
`membership_inference_attacks.py` contains the core attack code  
In each folder, `MIA_evaluate.py` performs attacks against target machine learning classifiers.  

If you want to compute the privacy risk score, first import `privacy_risk_score_utils.py`; after initializing the attack class in `MIA_evaluate.py`, add `risk_score = calculate_risk_score(MIA.s_tr_m_entr, MIA.s_te_m_entr, MIA.s_tr_labels, MIA.s_te_labels, MIA.t_tr_m_entr, MIA.t_tr_labels)`
