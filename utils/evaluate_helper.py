
#%%
import numpy as np
import torch
import requests
from sklearn.metrics import roc_auc_score, average_precision_score

def beta2M(beta_value):
    """
    beta value to m value, ref https://www.zxzyl.com/archives/1129/
    """
    return np.log2(beta_value/(1.0 - beta_value))


def M2beta(m_value):
    """
    m value to beta value, ref https://www.zxzyl.com/archives/1129/
    """
    return 2**m_value/(2**m_value + 1.0)



def beta2M_zx(beta_value, alpha = 0.001):
    """
    beta value to m value, use
    现有 zixin 采用的版本
    """
    return np.log((beta_value + alpha)/(1.0 - beta_value + alpha))


def M2beta_zx(m_value, alpha = 0.001):
    """
    m value to beta value, ref https://www.zxzyl.com/archives/1129/
    现有 zixin 采用的版本
    """

    print("#"*1000)
    return (np.e**m_value*(1+alpha) - alpha)/(np.e**m_value + 1.0 )


def plus_point_five(x):
    """
    handle pandas data, if interger values of age, then add 0.5
    """
    if(x - np.floor(x) < 1e-4):
        return x + 0.5
    else:
        return x


#Horvath's age transformation function
def anti_transform_age(exps):

    """
    not elegent!
    refer tp Evaluation of different computational methods for DNA methylation-based biological age
    implement by AltumAge
    """
    adult_age = 20
    ages = []
    for exp in exps:
        if exp < 0:
            age = (1 + adult_age)*(np.exp(exp))-1
            ages.append(age)
        else:
            age = (1 + adult_age)*exp + adult_age
            ages.append(age)
    ages = np.array(ages)
    return ages


def forward_transform_age(exps):
    """
    refer tp Evaluation of different computational methods for DNA methylation-based biological age
    implement by yichen
    """
    adult_age = 20
    ages = []
    for exp in exps:
        if exp <= adult_age:
            age = np.log(exp + 1.0) - np.log(adult_age +1)
            ages.append(age)
        else:
            age = (exp - adult_age)/(adult_age + 1.0)
            ages.append(age)
    ages = np.array(ages)
    return ages



def cal_torch_mae(torch_a, torch_b):
    """
    输入维度 [N*1]
    
    """
    value = torch.mean(torch.abs(torch_a - torch_b))
    return value.cpu().item()


def cal_torch_rmse(torch_a, torch_b):
    value = torch.sqrt(torch.mean(torch.square( torch_a - torch_b)))
    return value.cpu().item()


def cal_torch_MedAe(torch_a, torch_b):
    value = torch.median(torch.abs(torch_a - torch_b))
    return value.cpu().item()


def cal_torch_Rvalue(torch_a, torch_b):
    """
    [2,N] can be applied to  torch.corrcoef
    """

    concat_dim1 = torch.concat([torch_a, torch_b], axis = 1)
    concat_dim0 = concat_dim1.permute(1,0)
    corrvalue = torch.corrcoef(concat_dim0)
    return corrvalue[0,1].cpu().item()
    

def methylage_evaluate_score(predicts, targets):
    """ 
    shpae N*1
    """
    mae_value = cal_torch_mae(predicts, targets)
    rmse_value = cal_torch_rmse(predicts, targets)
    R_value = cal_torch_Rvalue(predicts, targets)
    medae_value = cal_torch_MedAe(predicts, targets)

    return [mae_value, rmse_value, R_value,  medae_value]


def classification_eval(predictions, labels):
    roc_auc = roc_auc_score(
        labels.cpu().detach().numpy(), predictions.cpu().detach().numpy()[:, 1])
    prc_auc = average_precision_score(
        labels.cpu().detach().numpy(), predictions.cpu().detach().numpy()[:, 1])

    predictions = predictions.argmax(dim=1)
    confusion_matrix = torch.zeros(2, 2)
    for p, l in zip(predictions.view(-1), labels.view(-1)):
        confusion_matrix[p, l] += 1
    # Accuracy
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / confusion_matrix.sum()

    # Precision
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])

    # Recall
    recall = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

    # F1-Score
    f1 = 2 * precision * recall / (precision + recall)

    # AUROC
    # roc_auc_score(predictions.cpu().detach().numpy(), torch.cat(list_pred)[:,1].cpu().detach().numpy())
    return accuracy, precision, recall, f1, roc_auc, prc_auc


def msg(text):
    webhook = 'https://open.feishu.cn/open-apis/bot/v2/hook/505931b6-2ff7-4457-8b08-97f4832d06bf'
    header = {
        "Content-Type": "application/json;charset=UTF-8"
    }
    message_body = {
        "msg_type": "text",
        "content": {
            "text": text
        }
    }
    ChatRob = requests.post(url=webhook, 
                            json=message_body, 
                            headers=header)





# %%

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


def calculate_cor_p(x,y):
    # Generate some random data for the example
    # Add a constant term to the predictor variable (required for linear regression)
    X = sm.add_constant(x)

    # Fit a linear regression models and print the summary
    model = sm.OLS(y, X).fit()
    # print(models.summary())

    # Extract the p-value for the slope coefficient
    corr = np.sqrt(model.rsquared)
    p_value = model.pvalues[1]

    return corr, p_value

# %%


#%%
if __name__ == "__main__":
    value = np.asarray([35.,19.0, 1.0])
    result = forward_transform_age(value)
    value_inverse = anti_transform_age(result)

    print(result.shape)
    print(result)
    print(value_inverse)

    beta_value = np.asarray([0.00, 0.2, 0.4, 0.6, 0.9, 1.0 ])
    m_value = beta2M_zx(beta_value)
    tt = M2beta_zx(m_value)
    print(m_value)
    print(tt)
    msg(f"I am dawin")



    x = np.random.rand(100)
    y = 0.7 * x + np.random.rand(100)
    corr, p_value = calculate_cor_p(x, y)
    print(corr, p_value)
# %%


