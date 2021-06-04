from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import fbeta_score

#Function that contains all the required metrics
def metrics(actual,predicted):
    precision = precision_score(actual,predicted)
    recall = recall_score(actual,predicted)
    fscore = fbeta_score(actual, predicted, beta=1, zero_division=0)
    accuracy = np.sum(actual == predicted) / len(predicted)
    f_b_p_score = fbeta_score(actual, predicted, average='micro', beta=0.5)
    f_b_r_score = fbeta_score(actual, predicted, average='micro', beta=1.5)
    return precision,recall,fscore,accuracy,f_b_p_score,f_b_r_score