# @title PerceiverIO
# https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_io.py
# https://arxiv.org/pdf/2107.14795
import torch
from torch import nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class Attention(nn.Module):
    # def __init__(self, d_model, cond_dim=None, n_heads=None, d_head=8, dropout=0.): # .1
    def __init__(self, query_dim, cond_dim=None, n_heads=8, d_head=64, drop=0):
        super().__init__()
        d_model = d_head * n_heads
        self.d_head, self.n_heads = d_head, n_heads
        # self.n_heads = d_model // d_head
        # self.d_head = d_model // n_heads
        self.cond_dim = cond_dim
        self.q = nn.Linear(query_dim, d_model, bias=False)
        self.kv = nn.Linear(cond_dim or query_dim, 2*d_model, bias=False)
        # self.lin = nn.Linear(d_model, d_model)
        self.lin = zero_module(nn.Linear(d_model, d_model))
        self.drop = nn.Dropout(drop) # indp before q,k,v; after linout
        self.scale = self.d_head**-.5

    def forward(self, x, cond=None, mask=None): # [batch, T, d_model]=[batch, h*w, c], [batch, num_tok, cond_dim], [batch,T]
        if self.cond_dim==None: cond=x # is self attn
        q = self.q(x).unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2) # [batch, T, d_model] -> [batch, n_heads, T, d_head]
        k, v = self.kv(cond).unflatten(-1, (self.n_heads, 2*self.d_head)).transpose(1, 2).chunk(2, dim=-1) # [batch, n_heads, T/num_tok, d_head]

        # # linear attention # Softmax(q) @ (Softmax(k).T @ v)
        if mask != None:
            mask = mask[:, None, :, None] # [batch,T] -> [batch,1,T,1]
            k, v = k.masked_fill(mask, -torch.finfo(x.dtype).max), v.masked_fill(mask, -torch.finfo(x.dtype).max)
        q, k = q.softmax(dim=-1)*self.scale, k.softmax(dim=-2)
        context = k.transpose(-2,-1) @ v # [batch, n_heads, d_head, d_head]
        out = q @ context # [batch, n_heads, T/num_tok, d_head]

        # # (quadratic) attention # Softmax(q @ k.T) @ v
        # attn = q @ k.transpose(-2,-1) * self.scale # [batch, n_heads, T] # [batch, n_heads, T, T/num_tok]
        # if mask != None: attn = attn.masked_fill(mask[:, None, :, None], -torch.finfo(attn.dtype).max) # [batch,T]->[batch,1,T,1]
        # attention = torch.softmax(attn, dim=-1)
        # out = self.drop(attention) @ v # [batch, n_heads, T, d_head]

        out = out.transpose(1, 2).flatten(2)
        return self.drop(self.lin(out)) # [batch, T, d_model]


class AttentionBlock(nn.Module):
    # def __init__(self, d_model, cond_dim=None, d_head, ff_dim=None, dropout=0.):
    def __init__(self, d_model, n_heads ,cond_dim=None, ff_dim=None, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.cond_dim = cond_dim
        self.norm1 = nn.RMSNorm(d_model) # LayerNorm RMSNorm
        if cond_dim!=None: self.norm2 = nn.RMSNorm(cond_dim)
        self.drop = nn.Dropout(dropout)
        self.attn = Attention(d_model, cond_dim, n_heads=n_heads, d_head=d_model//n_heads)
        act = nn.ReLU()
        if ff_dim==None: ff_dim=d_model*4
        self.ff = nn.Sequential(
            nn.RMSNorm(d_model), act, nn.Linear(d_model, ff_dim),
            nn.RMSNorm(ff_dim), act, zero_module(nn.Linear(ff_dim, d_model))
        )

    def forward(self, x, cond=None, mask=None): # [b,c,h,w], [batch, num_tok, cond_dim], [batch,T]
        if self.cond_dim==None: x = x + self.attn(self.norm1(x), mask)
        else: x = x + self.attn(self.norm1(x), self.norm2(cond), mask) # maybe no res for decoder
        x = x + self.ff(x) # maybe no ff for decoder?
        return x


class PerceiverIO(nn.Module):
    def __init__(self, in_dim, query_dim, out_dim=None, depth=12, num_latents = 512, latent_dim=512,
        cross_heads=1, latent_heads=8):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        # self.query = nn.Parameter(torch.randn(1, 1, query_dim))
        self.encode = AttentionBlock(latent_dim, cross_heads, cond_dim=in_dim)
        self.process = nn.Sequential(*[AttentionBlock(latent_dim, latent_heads) for _ in range(depth)])
        self.decode = AttentionBlock(query_dim, cross_heads, cond_dim=latent_dim)
        self.out = nn.Linear(query_dim, out_dim) if out_dim else nn.Identity()

    def forward(self, x, latent=None, query=None, mask=None): # [b,t,dim] , [1/b, num_latents, latent_dim], [1/b, query_len, query_dim], [b,t]
        latent = latent or self.latent
        latent = self.encode(latent, cond=x, mask=mask)
        latent = self.process(latent)
        # if query==None: return latent
        # query = query or self.query
        if query.ndim == 2: query = query.unsqueeze(0)
        out = self.decode(query, cond=latent)
        return self.out(out), latent

in_dim=1024
query_dim=8
# model = PerceiverIO(in_dim, query_dim, out_dim = None, depth=2, num_latents=3, latent_dim=6, cross_heads=1, latent_heads=2)
model = PerceiverIO(in_dim, query_dim, out_dim = None, depth=4, num_latents=32, latent_dim=256, cross_heads=4, latent_heads=2)
# cross_heads | in/query/out_dim
x = torch.rand(128,7,in_dim)
c=torch.rand(1,3,query_dim)
out, latent = model(x, query=c)
print(latent.shape)
print(out.shape) # [batch, query_len, query_dim]
