import torch
import timm
import numpy as np
from preprocess import preprocess



#  MobileOne
# using pretrained, and only embedding with global average pooling layer

model = timm.create_model('mobileone_s0', pretrained=True, num_classes=0, global_pool='avg')

# in eval mode
model.eval() 

def get_embedding(img):
  
    pre = preprocess(img)  

    #hwc->chw (nn with float)
    #unsqueeze(1 image at a point)
    tensor = torch.from_numpy(pre.transpose(2,0,1)).unsqueeze(0).float()
    
    with torch.no_grad(): #no gradient
        emb = model(tensor) #512
        emb = torch.nn.functional.normalize(emb, p=2, dim=1) #normalize_l2,direction only
    
    return emb.cpu().numpy()[0] #tensor to array


