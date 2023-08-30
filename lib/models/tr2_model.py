import os
import numpy as np
import torch
import torch.nn as nn
import lib.models.clip as clip
from lib.word_vectors import obj_edge_vectors
from lib.models.object_classifier import ObjectClassifier
from PIL import Image
from lib.models.feature_fusion import RelFeatFusion

class TR2(nn.Module):
    def __init__(self, mode, attention_class_num, spatial_class_num, contact_class_num, 
                 obj_classes, rel_classes, enc_layer_num, dec_layer_num, pre_path=''):
        super(TR2, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.pre_path=pre_path

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)

        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='./data/glove/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # relation feature fusion
        transformer_dim = self.subj_fc.out_features+self.obj_fc.out_features\
            + self.obj_embed.embedding_dim + self.obj_embed2.embedding_dim
        self.rff = RelFeatFusion(
            enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num,
            embed_dim=transformer_dim+512*1, nhead=8, dim_feedforward=2048, dropout=0.1)

        # classifier
        clip_sentence_path = './data/clip_sentence.npy'
        self.clip_sentence = torch.tensor(np.load(clip_sentence_path)).float().cuda()

        classifier_dim = (transformer_dim+512)*2
        if self.mode=='sgdet':
            self.a_rel_compress = nn.Linear(classifier_dim, self.attention_class_num)
            self.s_rel_compress = nn.Linear(classifier_dim, self.spatial_class_num)
            self.c_rel_compress = nn.Linear(classifier_dim, self.contact_class_num)
        else:
            self.a4 = nn.Linear(classifier_dim, 512)
            self.s4 = nn.Linear(classifier_dim, 512)
            self.c4 = nn.Linear(classifier_dim, 512)

        # visual diff
        self.change_adiff = nn.Linear(classifier_dim, 512)
        self.change_sdiff = nn.Linear(classifier_dim, 512)
        self.change_cdiff = nn.Linear(classifier_dim, 512)

        self.clip, self.clip_process = clip.load("ViT-B-32", device='cuda:0')

    def getVisualDiff(self, entry, global_output, ifTest=False):
        obj_idx_fea=entry['pair_idx'][:, 1] # index of obj in ent_feature
        obj_label=entry['pred_labels'][obj_idx_fea] # label of obj
        obj_label_unique, obj_label_cnt = torch.unique(obj_label,return_counts=True)
        obj_idx_dict={}
        for i,ol in enumerate(obj_label_unique):
            obj_idx_dict[int(ol)]=i
        n = len(obj_label_unique)
        t = max(int(entry['im_idx'][-1] + 1),int(obj_label_cnt.max()))  # number of frames
        l_cnt=[0 for _ in range(n)]
        idx_back=[[] for _ in range(n)]
        if not ifTest:
            a_label, s_label, c_label = \
                [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]

        diff_input = torch.zeros([n, t, global_output.shape[-1]]).to(global_output.device)
        for i,ol in enumerate(obj_label):
            obj_idx=obj_idx_dict[int(ol)]
            diff_input[obj_idx][l_cnt[obj_idx]] = global_output[i]
            if not ifTest:
                a_label[obj_idx].append(entry['attention_gt'][i])
                s_label[obj_idx].append(entry['spatial_gt'][i])
                c_label[obj_idx].append(entry['contacting_gt'][i])
            l_cnt[obj_idx]+=1
            idx_back[obj_idx].append(i)
        
        diff1 = diff_input[:,1:,:] - diff_input[:,:-1,:] # temporal difference
        diff_list = []
        for i, li in enumerate(l_cnt):
            diff_list.append(diff1[i][:li-1 ])
        diff_flatten = torch.cat(diff_list, 0) # n_rel - n

        a_change_logits1 = self.change_adiff(diff_flatten)
        s_change_logits1 = self.change_sdiff(diff_flatten)
        c_change_logits1 = self.change_cdiff(diff_flatten)

        if not ifTest:
            return [a_change_logits1, s_change_logits1, c_change_logits1], obj_label_unique, \
                [a_label, s_label, c_label]
        else:
            return [a_change_logits1, s_change_logits1, c_change_logits1], obj_label_unique, None

    def getTextDiff(self, obj_label_unique, rel_label_asc):
        all_a_diff, all_s_diff, all_c_diff = [],[],[]
        for i, ol in enumerate(obj_label_unique):
            a_label, s_label, c_label = rel_label_asc[0][i], rel_label_asc[1][i], rel_label_asc[2][i]
            assert len(a_label)==len(s_label)==len(c_label), \
                'different number of label %d %d %d'%(len(a_label),len(s_label),len(c_label))
            one_a_diff = torch.zeros((len(a_label), self.clip_sentence.shape[-1]),device=ol.device)
            one_s_diff, one_c_diff = torch.zeros_like(one_a_diff), torch.zeros_like(one_a_diff)
            for j, (a,s,c) in enumerate(zip(a_label,s_label, c_label)):
                one_a_diff[j]=self.clip_sentence[ol][a].mean(0,keepdim=True)
                one_s_diff[j]=self.clip_sentence[ol][s].mean(0,keepdim=True)
                one_c_diff[j]=self.clip_sentence[ol][c].mean(0,keepdim=True)
            all_a_diff.append(one_a_diff[1:]-one_a_diff[:-1]) # temporal difference
            all_s_diff.append(one_s_diff[1:]-one_s_diff[:-1])
            all_c_diff.append(one_c_diff[1:]-one_c_diff[:-1])
        all_a_diff = torch.cat(all_a_diff,0)
        all_s_diff = torch.cat(all_s_diff,0)
        all_c_diff = torch.cat(all_c_diff,0)
        return [all_a_diff, all_s_diff, all_c_diff]

    def getImageInput(self, all_input, bbox, im):
        part_im = im[bbox[1]:bbox[3],bbox[0]:bbox[2]]
        part_im_input = self.clip_process(Image.fromarray(part_im)).unsqueeze(0).to(bbox.device)
        all_input = torch.cat((all_input, part_im_input), 0)
        return all_input
    
    def getUnionFeat(self, entry):
        n_rel = len(entry['im_idx'])
        gpu_device = entry['im_idx'].device
        myunion_box = (entry['union_box']/entry['im_info'][0,2])[:,1:].int()
        all_union_input = torch.tensor([],device=gpu_device)
        for j in range(n_rel):
            im = entry['origin_ims'][entry['im_idx'][j].int()]
            all_union_input=self.getImageInput(all_union_input, myunion_box[j], im)
        union_feature = self.clip.encode_image(all_union_input)
        return union_feature.float()

    def forward(self, entry, ifTest=False):
        entry = self.object_classifier(entry)

        if self.mode=='predcls' and os.path.exists(self.pre_path):
            union_feat = torch.tensor(np.load(
                self.pre_path+'%s_img.npy'%entry['video_name'])).cuda()
        else:
            union_feat = self.getUnionFeat(entry)

        # visual part
        subj_rep = entry['features_1d'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features_1d'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        x_visual = torch.cat((subj_rep, obj_rep), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        # relation representation
        rel_features = torch.cat((union_feat,x_visual, x_semantic), dim=1)

        # Relation Feature Fusion (Implicitly)
        obj_idx_fea=entry['pair_idx'][:, 1] # index of obj in ent_feature
        obj_label=entry['pred_labels'][obj_idx_fea] # obj label
        if not ifTest:
            trans_output = self.rff(rel_features, entry['im_idx'], obj_label)
        else:
            trans_output = self.rff(rel_features, entry['im_idx'], obj_label)

        if not ifTest:
            diff_v, obj_label, rel_label_asc = self.getVisualDiff(entry, trans_output, ifTest)
            diff_t = self.getTextDiff(obj_label, rel_label_asc)

        text_feature = self.clip_sentence[obj_class].transpose(1,2) # as classifier
        at, st, ct = text_feature[:,:,:3], text_feature[:,:,3:9], text_feature[:,:,9:]

        if self.mode=='sgdet':
            entry["attention_distribution"] = self.a_rel_compress(trans_output)
            entry["spatial_distribution"] = self.s_rel_compress(trans_output)
            entry["contacting_distribution"] = self.c_rel_compress(trans_output)
        else:
            a4, s4, c4 = self.a4(trans_output), self.s4(trans_output), self.c4(trans_output)
            entry["attention_distribution"] = torch.matmul(a4.unsqueeze(1),at).squeeze(1)
            entry["spatial_distribution"] = torch.matmul(s4.unsqueeze(1),st).squeeze(1)
            entry["contacting_distribution"] = torch.matmul(c4.unsqueeze(1),ct).squeeze(1)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])

        if not ifTest:
            return entry, diff_v, diff_t
        else:
            return entry, None, None
