import torch
import torch.nn.functional as nnf
import abc
import math
from torchvision.utils import save_image


LOW_RESOURCE = False
MAX_NUM_WORDS = 77
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def start_att_layers(self):
        return self.start_ac_layer #if LOW_RESOURCE else 0
    @property
    def end_att_layers(self):
        return self.end_ac_layer
    
    
    

    @abc.abstractmethod
    def forward(self, q, k, v, num_heads,attn):
        raise NotImplementedError

    def attn_forward(self, q, k, v, num_heads,attention_probs,attn):
        if q.shape[0]//num_heads == 3:
            h_s_re = self.forward(q, k, v, num_heads,attention_probs, attn)
            
        else:
            uq,cq = q.chunk(2)
            uk,ck = k.chunk(2)
            uv,cv = v.chunk(2)
            u_attn, c_attn = attention_probs.chunk(2)
            
            u_h_s_re = self.forward(uq, uk, uv, num_heads,u_attn, attn)

            c_h_s_re = self.forward(cq, ck, cv, num_heads,c_attn, attn)
            h_s_re = (u_h_s_re, c_h_s_re)
        return h_s_re
    
    def __call__(self, q, k, v, num_heads,attention_probs,attn):

        
        if self.cur_att_layer >= self.start_att_layers and self.cur_att_layer < self.end_att_layers:
            h_s_re = self.attn_forward(q, k, v, num_heads,attention_probs,attn)
        else:
            h_s_re=None
        
        
        self.cur_att_layer += 1
        
        if self.cur_att_layer == self.num_att_layers // 2: #+ self.num_uncond_att_layers:
            self.cur_att_layer = 0 #self.num_uncond_att_layers
            self.cur_step += 1
            self.between_steps()
        return h_s_re

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

def enhance_tensor(tensor: torch.Tensor, contrast_factor: float = 1.67) -> torch.Tensor:
    """ Compute the attention map contrasting. """
    mean_feat = tensor.mean(dim=-1, keepdims=True)
    adjusted_tensor = (tensor - mean_feat) * contrast_factor + mean_feat
    return adjusted_tensor

class AttentionStyle(AttentionControl):

    def __init__(self, 
                 num_steps,
                 start_ac_layer, end_ac_layer,
                 style_guidance=0.3,
                 mix_q_scale=1.0,
                 de_bug=False,
                 ):
        super(AttentionStyle, self).__init__()


        self.start_ac_layer = start_ac_layer
        self.end_ac_layer = end_ac_layer
        self.num_steps=num_steps
        self.de_bug = de_bug
        self.style_guidance = style_guidance
        self.coef = None
        self.mix_q_scale = mix_q_scale
        
    def forward(self, q, k, v, num_heads, attention_probs, attn):

        
        if self.de_bug:
                import pdb; pdb.set_trace()

        if self.mix_q_scale < 1.0:
            q[num_heads*2:] = q[num_heads*2:] * self.mix_q_scale + (1 - self.mix_q_scale) * q[num_heads*1:num_heads*2]
        b,n,d = k.shape
        re_q = q[num_heads*2:] # b,n,d,
        re_k = torch.cat([k[num_heads*1:num_heads*2],k[num_heads*0:num_heads*1]],dim=1) #b,2n,d
        v_re = torch.cat([v[num_heads*1:num_heads*2],v[num_heads*0:num_heads*1]],dim=1) #b,2n,d
        re_sim = torch.bmm(re_q, re_k.transpose(-1, -2)) * attn.scale
        re_sim[:,:,n:] = re_sim[:,:,n:] * self.style_guidance
        re_attention_map = re_sim.softmax(-1)
        h_s_re = torch.bmm(re_attention_map, v_re)



        return h_s_re
        

    


