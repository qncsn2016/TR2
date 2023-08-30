import torch
import torch.nn as nn
from lib.models.utils import _get_clones, get_timing_signal_1d

class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1, attn_mask=False):
        super().__init__()

        self.multihead2 = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.nhead = nhead
        self.attn_mask = attn_mask

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, global_input, input_key_padding_mask, position_embed):
        if position_embed is not None:
            q = global_input+position_embed
            k = global_input+position_embed
            v = global_input
        else:
            q = global_input
            k = global_input
            v = global_input
        if self.attn_mask:
            amask=torch.triu(torch.ones((q.shape[1], q.shape[1])),diagonal=1).bool().to(q.device)
            tgt2, global_attention_weights = self.multihead2(q, k, v,
                key_padding_mask=input_key_padding_mask, attn_mask=amask)
        else:
            tgt2, global_attention_weights = self.multihead2(q, k, v,
                key_padding_mask=input_key_padding_mask)

        tgt = global_input + self.dropout2(tgt2)
        tgt = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt, global_attention_weights

class TransformerDecoder(nn.Module):
    
    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, global_input, input_key_padding_mask, position_embed):
        output = global_input
        for i, layer in enumerate(self.layers):
            output, _ = layer(output, input_key_padding_mask, position_embed)
        return output 

class TransformerEncoderLayer(nn.Module):

    def __init__(self, embed_dim=1936, nhead=4, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, input_key_padding_mask):
        # local attention
        src2, local_attention_weights = self.self_attn(src, src, src, key_padding_mask=input_key_padding_mask)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(nn.functional.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, local_attention_weights

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, input, input_key_padding_mask):
        output = input
        weights = torch.zeros([self.num_layers, output.shape[1], output.shape[0], output.shape[0]]).to(output.device)

        for i, layer in enumerate(self.layers):
            output, local_attention_weights = layer(output, input_key_padding_mask)
            weights[i] = local_attention_weights
        if self.num_layers > 0:
            return output, weights
        else:
            return output, None

class RelFeatFusion(nn.Module):
    def __init__(self, enc_layer_num, dec_layer_num,
            embed_dim=1936, nhead=8, dim_feedforward=2048, dropout=0.1):
        super(RelFeatFusion, self).__init__()

        # spatial transformer encoder
        encoder_layer = TransformerEncoderLayer(embed_dim=embed_dim, nhead=nhead, \
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = TransformerEncoder(encoder_layer, enc_layer_num)

        # temporal transformer decoder
        decoder_layer = TransformerDecoderLayer(embed_dim=embed_dim, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout, attn_mask=False)
        self.decoder=TransformerDecoder(decoder_layer, dec_layer_num)

        # short-term message token
        self.diff_two_fc = nn.Linear(embed_dim*2, embed_dim)
        self.att_fc1 = nn.Linear(embed_dim, embed_dim)
        self.att_fc2 = nn.Linear(embed_dim, embed_dim)
        self.tanh = nn.Tanh()      

    def forward(self, features, im_idx, obj_label):
        l = torch.sum(im_idx == torch.mode(im_idx)[0])  # the highest box number in the single frame
        # b = int(im_idx[-1] + 1)
        obj_label_unique, obj_label_cnt=torch.unique(obj_label,return_counts=True)
        b = max(int(im_idx[-1] + 1),int(obj_label_cnt.max()))
        encoder_input = torch.zeros([l, b, features.shape[1]]).to(features.device)
        masks = torch.zeros([b, l], dtype=torch.uint8).to(features.device)
        # Padding/Mask maybe don't need for-loop
        for i in range(b):
            encoder_input[:torch.sum(im_idx == i), i, :] = features[im_idx == i]
            masks[i, torch.sum(im_idx == i):] = 1

        # spatial encoder
        encoder_output, _ = self.encoder(encoder_input, masks)
        encoder_output = (encoder_output.permute(1, 0, 2)).contiguous().view(
            -1, features.shape[1])[masks.view(-1) == 0]

        obj_idx_dict={}
        for i,ol in enumerate(obj_label_unique):
            obj_idx_dict[int(ol)]=i     
        n = len(obj_label_unique)

        # Rearrangement by time
        r1 = torch.zeros([n, b, features.shape[-1]]).to(features.device)
        l_cnt=[0 for _ in range(n)]
        idx_back=[[] for _ in range(n)]
        for i,ol in enumerate(obj_label):
            obj_idx=obj_idx_dict[int(ol)]
            r1[obj_idx][l_cnt[obj_idx]] = encoder_output[i]
            l_cnt[obj_idx]+=1
            idx_back[obj_idx].append(i)
        r1 = r1[:,:max(l_cnt),:]

        # temporal transformer decoder
        mask = torch.ones([n, max(l_cnt)],dtype=torch.uint8, device=features.device)
        for i in range(n):
            mask[i][:l_cnt[i]]=0
        pos_emb = get_timing_signal_1d(max(l_cnt), r1.shape[-1]).unsqueeze(0).to(features.device)
        decoder_output = self.decoder(r1 , mask.bool(), pos_emb)

        # short-term message token
        d0 = torch.cat((torch.zeros_like(decoder_output[:,:1,:]),decoder_output[:,:-1,:]), 1)
        d1 = torch.cat((d0,decoder_output),-1)
        diff = self.diff_two_fc(d1)
        diff1 = self.tanh(self.att_fc1(diff))
        diff2 = self.tanh(self.att_fc2(diff1))
        
        diff_mul = torch.cat((decoder_output[:,:1,:], decoder_output[:,:-1,:]), 1)
        token_output = torch.cat((decoder_output, diff2*diff_mul), -1)
        
        # Rearrangement by frame
        output = torch.zeros([features.shape[0], features.shape[1]*2], device=features.device)
        for i in range(n):
            for n in range(l_cnt[i]):
                origin_idx=idx_back[i][n]
                output[origin_idx] = token_output[i][n]
        
        return output

