import torch
import numpy as np
from dataloader.action_genome import AG
import copy
import lib.models.clip as clip

# load name
AG_dataset_train = AG(mode="train", datasize='mini', 
    data_path='data/action_genome/', filter_nonperson_box_frame=True, filter_small_box=False)
obj_name = copy.copy(AG_dataset_train.object_classes)
rel_name = AG_dataset_train.relationship_classes

# correct name
a={9:'closet or cabinet', 11:'cup or glass or bottle', 23:'notebook or paper', 24:'phone or camera', 31:'sofa or couch'}
for k,v in a.items():
    obj_name[k]=v

new_name = copy.copy(rel_name)
b={0:'looking at',1:'not looking at',5:'in front of',7:'on the side of',10:'covered by',
    11:'drinking from', 13:'having it on the back', 15:'leaning on', 16:'lying on',
    17:'not contacting', 18:'and', 19:'sitting on', 20:'standing on', 25:'writing on'}
for k,v in b.items():
    new_name[k]=v

# relation-aware prompting
all_sent=[]
for o in obj_name:
    obj_sent=[]
    for ri, r in enumerate(new_name):
        if ri >=3 and ri <=7:
            obj_sent.append('a photo of a %s %s a person'%(o,r))
        elif r=='having it on the back':
            obj_sent.append('a photo of a person having a %s on the back'%(o))
        else:
            obj_sent.append('a photo of a person %s a %s'%(r,o))
    all_sent.append(obj_sent)

# pretrained text encoder
myclip, _ = clip.load("ViT-B-32", device='cuda:0')
res=[]
with torch.no_grad():
    for obj_sent in all_sent:
        obj_token = clip.tokenize(obj_sent).cuda()
        obj_feat = myclip.encode_text(obj_token)[None,...]
        res.append(obj_feat)
np.save('new_clip_sentence.npy',torch.cat(res,0).detach().cpu().numpy())