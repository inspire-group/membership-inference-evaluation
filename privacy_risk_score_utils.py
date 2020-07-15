import numpy as np
import matplotlib.pyplot as plt

def distrs_compute(tr_values, te_values, tr_labels, te_labels, num_bins=5, log_bins=True, plot_name=None):
    
    ### function to compute and plot the normalized histogram for both training and test values class by class.
    ### we recommand using the log scale to plot the distribution to get better-behaved distributions.
    
    num_classes = len(set(tr_labels))
    sqr_num = np.ceil(np.sqrt(num_classes))
    tr_distrs, te_distrs, all_bins = [], [], []
    
    plt.figure(figsize = (15,15))
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', linewidth=2)
    
    for i in range(num_classes):
        tr_list, te_list = tr_values[tr_labels==i], te_values[te_labels==i]
        if log_bins:
            # when using log scale, avoid very small number close to 0
            small_delta = 1e-10
            tr_list[tr_list<=small_delta] = small_delta
            te_list[te_list<=small_delta] = small_delta
        n1, n2 = np.sum(tr_labels==i), np.sum(te_labels==i)
        all_list = np.concatenate((tr_list, te_list))
        max_v, min_v = np.amax(all_list), np.amin(all_list)
        
        plt.subplot(sqr_num, sqr_num, i+1)
        if log_bins:
            bins = np.logspace(np.log10(min_v), np.log10(max_v),num_bins+1)
            weights = np.ones_like(tr_list)/float(len(tr_list))
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
            plt.gca().set_xscale("log")
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
            plt.gca().set_xscale("log")
        else:
            bins = np.linspace(min_v, max_v,num_bins+1)
            weights = np.ones_like(tr_list)/float(len(tr_list))
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
        tr_distrs.append(h1)
        te_distrs.append(h2)
        all_bins.append(bins)
    if plot_name == None:
        plot_name='./tmp'
    plt.savefig(plot_name+'.png', bbox_inches='tight')
    tr_distrs, te_distrs, all_bins = np.array(tr_distrs), np.array(te_distrs), np.array(all_bins)
    return tr_distrs, te_distrs, all_bins


def risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels):
    
    ### Given training and test distributions (obtained from the shadow classifier), 
    ### compute the corresponding privacy risk score for training points (of the target classifier).
    
    def find_index(bins, value):
        # for given n bins (n+1 list) and one value, return which bin includes the value
        if value>=bins[-1]:
            return len(bins)-2 # when value is larger than any bins, we assign the last bin
        if value<=bins[0]:
            return 0  # when value is smaller than any bins, we assign the first bin
        return np.argwhere(bins<=value)[-1][0]
    
    def score_calculate(tr_distr, te_distr, ind): 
        if tr_distr[ind]+te_distr[ind] != 0:
            return tr_distr[ind]/(tr_distr[ind]+te_distr[ind])
        else: # when both distributions have 0 probabilities, we find the nearest bin with non-zero probability
            for t_n in range(1, len(tr_distr)):
                t_ind = ind-t_n
                if t_ind>=0:
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                t_ind = ind+t_n
                if t_ind<len(tr_distr):
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                    
    risk_score = []   
    for i in range(len(data_values)):
        c_value, c_label = data_values[i], data_labels[i]
        c_tr_distr, c_te_distr, c_bins = tr_distrs[c_label], te_distrs[c_label], all_bins[c_label]
        c_index = find_index(c_bins, c_value)
        c_score = score_calculate(c_tr_distr, c_te_distr, c_index)
        risk_score.append(c_score)
    return np.array(risk_score)

def calculate_risk_score(tr_values, te_values, tr_labels, te_labels, data_values, data_labels, 
                         num_bins=5, log_bins=True):
    
    ########### tr_values, te_values, tr_labels, te_labels are from shadow classifier's training and test data
    ########### data_values, data_labels are from target classifier's training data
    ########### potential choice for the value -- entropy, or modified entropy, or prediction loss (i.e., -np.log(confidence))
    
    tr_distrs, te_distrs, all_bins = distrs_compute(tr_values, te_values, tr_labels, te_labels, 
                                                    num_bins=num_bins, log_bins=log_bins)
    risk_score = risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels)
    return risk_score
