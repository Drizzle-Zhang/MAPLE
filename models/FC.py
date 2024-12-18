
import torch
import torch.nn as nn
import torch.nn.functional as F

class block(nn.Module):
    def __init__(self, input_size, output_size, 
    if_bn = False, if_dp = False):
        super(block, self).__init__()
        self.if_bn = if_bn
        self.if_dp = if_dp
        self.fc1 = nn.Linear(input_size, output_size)

        if self.if_bn:
            self.bn1 = nn.BatchNorm1d(output_size)
        self.relu = nn.ReLU()

        if self.if_dp:
            self.dropout = nn.Dropout(p = 0.2)


    def forward(self, x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        x = self.fc1(x)
        if self.if_bn:
            x = self.bn1(x)
        x = self.relu(x)

        if self.if_bn:
            x = self.bn1(x)
        if self.if_dp:
            x = self.dropout(x)
        return x


class FCNet(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_number, hidden_size,
                 if_bn = True, if_dp = True):

        super(FCNet, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.if_bn = if_bn

        self.first_layer = block(feature_channel, hidden_size, 
        if_bn = if_bn, if_dp = if_dp)

        layers = []
        for i in range(hidden_number):
            layers.append(block(hidden_size, hidden_size, 
            if_bn = if_bn, if_dp = if_dp))
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_size, output_channel)


    def forward(self, x):
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return x



class FCNet_H(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp):
        super(FCNet_H, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.hidden_list = hidden_list

        self.first_layer = block(feature_channel, hidden_list[0],if_bn, if_dp)

        layers = []
        for i in range(len(hidden_list) - 1):
            layers.append(block(hidden_list[i], hidden_list[i+1],if_bn, if_dp) )
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_list[-1], output_channel)


        self.tissue_embedding_model = nn.Embedding(num_embeddings = 32, 
                                 embedding_dim = 32 )
                                 
        self.sex_embedding_model = nn.Embedding(num_embeddings = 4, 
                                 embedding_dim = 32 )


    def forward(self, x):
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return x



class FCNet_H_class(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp):
        super(FCNet_H_class, self).__init__()
        self.feature_channel = feature_channel
        self.output_channel = output_channel
        self.hidden_list = hidden_list

        self.first_layer = block(feature_channel, hidden_list[0],if_bn, if_dp)

        layers = []
        for i in range(len(hidden_list) - 1):
            layers.append(block(hidden_list[i], hidden_list[i+1],if_bn, if_dp) )
        self.mediate_layer = nn.Sequential(*layers)

        self.last_layer = nn.Linear(
            hidden_list[-1], output_channel)


        self.tissue_embedding_model = nn.Embedding(num_embeddings = 32,
                                                   embedding_dim = 32 )

        self.sex_embedding_model = nn.Embedding(num_embeddings = 4,
                                                embedding_dim = 32 )


    def forward(self, x):
        x = self.first_layer(x)
        x = self.mediate_layer(x)
        x = self.last_layer(x)
        return F.log_softmax(x, dim=1)




class FCNet_Embedding(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding = False, if_norm = False):
        super(FCNet_Embedding, self).__init__()
        
    
        self.if_embedding = if_embedding
        self.if_norm = if_norm

        if(self.if_embedding):
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings = 32, 
                                    embedding_dim = embedding_dim )
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings = 4, 
                                    embedding_dim = embedding_dim )

            self.fc_model = FCNet_H(feature_channel + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 

        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

        
    def forward(self, feature, additional):
        if(self.if_embedding):
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)
            tissue_embedding = self.tissue_embedding_model(tissue_index)
            sex_embedding = self.sex_embedding_model(sex_index)
            # tissue_embedding = self.tissue_embedding_model(tissue_index) * 7.0
            # sex_embedding = self.sex_embedding_model(sex_index)* 7.0
            feature_concat = torch.concat([feature, tissue_embedding, sex_embedding], axis = 1)
            result = self.fc_model(feature_concat)

        else:
            result = self.fc_model(feature)
        
        """ 是否归一化的模块
        """
        if(self.if_norm):
            result = result / torch.sqrt(torch.norm(result, p = 1, dim = 0))
        return result





class FCNet_Embedding_Mask(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding = False, if_norm = False, limit_feature_number = 200):
        super(FCNet_Embedding_Mask, self).__init__()
        
    
        self.if_embedding = if_embedding
        self.if_norm = if_norm

        self.mask = torch.randn(feature_channel)
        self.mask.requires_grad = True
        self.sigmoid = torch.nn.Sigmoid()
        self.limit_feature_number = limit_feature_number


        if(self.if_embedding):
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings = 32, 
                                    embedding_dim = embedding_dim )
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings = 4, 
                                    embedding_dim = embedding_dim )

            self.fc_model = FCNet_H(feature_channel + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 

        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

        
    def forward(self, feature, additional):

        mask_vector = self.sigmoid(self.mask ) 
        vals, idx = mask_vector.topk(self.limit_feature_number)

        mask_vector = torch.zeros_like(feature)
        mask_vector[:,idx] = vals


        masked_feature = feature * mask_vector

        if(self.if_embedding):
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)

            tissue_embedding = self.tissue_embedding_model(tissue_index) 
            sex_embedding = self.sex_embedding_model(sex_index)

            # mask_vector = self.sigmoid(self.mask ) * self.limit_feature_number
            # masked_feature = feature * mask_vector
            feature_concat = torch.concat([masked_feature, tissue_embedding, sex_embedding], axis = 1)
            result = self.fc_model(feature_concat)

        else:
            result = self.fc_model(masked_feature)
        
        """ 是否归一化的模块
        """
        if(self.if_norm):
            result = result / torch.sqrt(torch.norm(result, p = 1, dim = 0))
        return result, mask_vector




class FCNet_Embedding_Mask2(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding = False, if_norm = False):
        super(FCNet_Embedding_Mask2, self).__init__()
        
    
        self.if_embedding = if_embedding
        self.if_norm = if_norm
        self.linear_down = 1024
        self.linear1 = nn.Linear(feature_channel, self.linear_down)

        if(self.if_embedding):
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings = 32, 
                                    embedding_dim = embedding_dim )
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings = 4, 
                                    embedding_dim = embedding_dim )

            self.fc_model = FCNet_H(self.linear_down + embedding_dim*2, output_channel,
                 hidden_list, if_bn, if_dp) 

        else:
            self.fc_model = FCNet_H(self.linear_down, output_channel,
                 hidden_list, if_bn, if_dp)

        
    def forward(self, feature, additional):

        feature_dimdown = self.linear1(feature)
        if(self.if_embedding):
            device = feature.device
            tissue_index = additional["tissue_index"].to(device)
            sex_index = additional["sex_index"].to(device)
            tissue_embedding = self.tissue_embedding_model(tissue_index)
            sex_embedding = self.sex_embedding_model(sex_index)
            # tissue_embedding = self.tissue_embedding_model(tissue_index) * 7.0
            # sex_embedding = self.sex_embedding_model(sex_index)* 7.0
            feature_concat = torch.concat([feature_dimdown, tissue_embedding, sex_embedding], axis = 1)
            result = self.fc_model(feature_concat)

        else:
            result = self.fc_model(feature_dimdown)
        
        """ 是否归一化的模块
        """
        if(self.if_norm):
            result = result / torch.sqrt(torch.norm(result, p = 1, dim = 0))
        return result



class FCNet_2Embedding(nn.Module):
    def __init__(self, feature_channel, output_channel,
                 hidden_list, if_bn, if_dp, if_embedding = False):
        super(FCNet_2Embedding, self).__init__()
        
    
        self.if_embedding = if_embedding

        if(self.if_embedding):
            embedding_dim = 32
            self.tissue_embedding_model = nn.Embedding(num_embeddings = 32, 
                                    embedding_dim = embedding_dim )
                                    
            self.sex_embedding_model = nn.Embedding(num_embeddings = 4, 
                                    embedding_dim = embedding_dim )

            self.fc_model = FCNet_H(feature_channel + embedding_dim * 4, output_channel,
                 hidden_list, if_bn, if_dp) 

        else:
            self.fc_model = FCNet_H(feature_channel, output_channel,
                 hidden_list, if_bn, if_dp)

        

    def forward(self, feature, additional1, additional2):
        if(self.if_embedding):
            device = feature.device

            tissue_index1 = additional1["tissue_index"].to(device)
            sex_index1 = additional1["sex_index"].to(device)
            tissue_embedding1 = self.tissue_embedding_model(tissue_index1) * 7.0
            sex_embedding1 = self.sex_embedding_model(sex_index1)* 7.0


            tissue_index2 = additional2["tissue_index"].to(device)
            sex_index2 = additional2["sex_index"].to(device)
            tissue_embedding2 = self.tissue_embedding_model(tissue_index2) * 7.0
            sex_embedding2 = self.sex_embedding_model(sex_index2)* 7.0

            feature_concat = torch.concat([feature, tissue_embedding1, sex_embedding1, \
                tissue_embedding2, sex_embedding2], axis = 1)
        
            result = self.fc_model(feature_concat)

        else:
            result = self.fc_model(feature)
        return result




class FCNet_1(nn.Module):
    def __init__(self, feature_channel, output_channel):
        super(FCNet_1, self).__init__()
        self.fc1 = nn.Linear(feature_channel, output_channel)

    def forward(self, x):
        x = self.fc1(x)
        return x



if __name__  == "__main__":
    # models = FCNet(feature_channel= 20, output_channel = 2,
    #         hidden_number= 1, hidden_size= 10)
    # x = torch.rand(128, 20)
    # y = models(x)
    # print(y.shape)

    # model2 = FCNet_H(feature_channel= 20, output_channel = 2, 
    #         hidden_list = [10,20,30,40], 
    #         if_bn= False, if_dp = False)
    # y = model2(x)
    # print(y.shape)

    # model3 = FCNet_1(feature_channel= 20, output_channel = 2)
    # y = model2(x)
    # print(y.shape)

    model = FCNet_Embedding(feature_channel = 20, output_channel =1,
                 hidden_list = [3,3,3],  if_bn = False, if_dp = False)
    x = torch.rand(128, 20)
    y = model(x,x)
    print(y.shape)


    # 用来测试限制 模型参数的 
    import torch.optim as optim

    model = FCNet_Embedding_Mask(feature_channel = 500, output_channel =1,
                 hidden_list = [3,3,3],  if_bn = False, if_dp = False, limit_feature_number = 100)
    x = torch.rand(128, 500)
    y, mask_vector = model(x, x)
    print(y.shape)

    optimizer = optim.Adam([{"params":model.parameters()},
                            {"params":model.mask}], 
                            lr =0.01)
    print(mask_vector.shape)
    print(mask_vector)
    print(torch.sum(mask_vector))

    for i in range(5000):
        y, mask_vector = model(x, x)
        loss = -torch.mean(torch.sum( mask_vector**2, dim =1 ))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(mask_vector[0,:])


#%%




# %%
