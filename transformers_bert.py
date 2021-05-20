import torch
import torch.nn as nn
from transformers import BertModel

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Linear(input_size, 768)        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.cls = nn.Linear(768, output_size)
    def forward(self, x):
        x = self.fc(x) #这里也可以加一个激活函数
        out = self.bert(inputs_embeds = x)
        out = self.cls(out[0])
        out = torch.mean(out, 1) #对每个特征求平均作为最终特征向量
        return out
        
x = torch.rand(2,100,14346)
model = MLP(14346,2)
out = model(x)
out.size()