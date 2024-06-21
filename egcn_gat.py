import utils as u
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch.nn import functional as F
import torch_geometric
import torch_scatter

class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cuda', skipfeats=False, gat=False):
        super().__init__()
        GRCU_args = u.Namespace({})
        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation,
                                     'device':device})
            grcu_i = GRCU(GRCU_args, gat, args.recurrent_unit)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

        model_parameters = filter(
            lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"EGCN-h parameter {params}")

    def parameters(self):
        return self._parameters

    def forward(self, A_list, Nodes_list, nodes_mask_list, edge_weights=None):
        node_feats = Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list, edge_weights)

        # out = Nodes_list
        # out = Nodes_list[-1]
        # if self.skipfeats:
        #     # use node_feats.to_dense() if 2hot encoded input
        #     out = torch.cat((out, node_feats), dim=1)
        out_sequence = Nodes_list
        return out_sequence


class GRCU(torch.nn.Module):
    def __init__(self,args, gat=True, recurrent_unit="gru"):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        hidden_size =  args.in_feats*args.out_feats
        self.recurrent_unit = recurrent_unit

        # Setting the recurrent unit based on input arguments
        if recurrent_unit == "gru":
            self.evolve_weights = torch.nn.GRUCell(args.in_feats, hidden_size)
        elif recurrent_unit == "lstm":
            self.evolve_weights = torch.nn.LSTMCell(args.in_feats, hidden_size)

        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.cell_state_init = torch.zeros((self.GCN_init_weights.flatten().shape)).to(self.args.device)

        self.reset_param(self.GCN_init_weights)

        # Setting the convolutional layer based on input arguments
        if gat:
            self.conv = GAT_MP(out_channels=self.args.out_feats, device=self.args.device)
        else:
            self.conv = MP(in_channels = self.args.in_feats, out_channels=self.args.out_feats)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list, edge_weights):
        GCN_weights = self.GCN_init_weights
        cell_state = self.cell_state_init
        out_seq = []
        for t, Ahat in enumerate(A_list):

            mask = mask_list[t].flatten()
            node_embs = node_embs_list[t].to_dense()

            input_GRU = torch.sum(torch.mul(torch.softmax(mask, dim=0), node_embs.t()), axis=1)
            hidden_GRU = GCN_weights.flatten()

            if self.recurrent_unit == "gru":
                GCN_weights = self.evolve_weights(input_GRU, hidden_GRU)
            elif self.recurrent_unit == "lstm":
                GCN_weights, cell_state = self.evolve_weights(input_GRU, (hidden_GRU, cell_state))

            GCN_weights = GCN_weights.reshape(self.GCN_init_weights.shape)
            node_embs = self.conv(node_embs, weights=GCN_weights)
            node_embs = self.args.activation(node_embs)

            out_seq.append(node_embs)

        return out_seq

class MP(MessagePassing):

    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):
        super(MP, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize


    def forward(self, x, edge_index, weights, size = None, edge_weights=None):

        # Using weights calculated in the recurrent unit layer
        out = self.propagate(edge_index, size=size, x = (x, x))
        # Update our node embedding with skip connection from the previous layer
        skip = x.matmul(weights)

        out = out.matmul(weights) + skip

        if self.normalize:
          out = F.normalize(out)

        return out

    def message(self, x_j):

        out = x_j

        return out

    def aggregate(self, inputs, index, dim_size = None):

        node_dim = self.node_dim

        # Performing mean aggregation
        out = torch_scatter.scatter(inputs.to_dense(), index.to_dense(), node_dim, reduce="mean")

        return out

class GAT_GCN(torch.nn.Module):
    def __init__(self, activation, in_features, out_features):
        ## Implementation in pure torch for completeness
        super().__init__()
        self.activation = activation
        attention_dim = 2
        self.w = Parameter(torch.Tensor(in_features, out_features))
        self.w_a = Parameter(torch.Tensor(out_features, attention_dim))
        self.a = Parameter(torch.Tensor(2*attention_dim, 1))

        u.reset_param(self.w)
        u.reset_param(self.a)
        u.reset_param(self.w_a)
        self.alpha = 0.001

    def forward(self, node_feats, Ahat):
        N = Ahat.shape[0]
        h_prime = node_feats.matmul(self.w)
        h_reduced = h_prime.matmul(self.w_a)
        H1 = h_reduced.unsqueeze(1).repeat(1,N,1)
        H2 = h_reduced.unsqueeze(0).repeat(N,1,1)
        attn_input = torch.cat([H1, H2], dim = -1) # (N, N, F)
        e = attn_input.matmul(self.a).squeeze(-1) # [N, N]
        attn_mask = -1e18*torch.ones_like(e)
        masked_e = torch.where(Ahat.to_dense() > 0, e, attn_mask)
        attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

        h_prime = torch.mm(attn_scores, h_prime)
        out = self.activation(h_prime)
        return out
    
class GAT_MP(MessagePassing):

    def __init__(self, out_channels, device, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT_MP, self).__init__(node_dim=0, **kwargs)

        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.device = device

        self.att_l = Parameter(torch.zeros((1, heads, out_channels//heads)))
        self.att_r = Parameter(torch.zeros((1, heads, out_channels//heads)))
        self.reset_parameters()


    def reset_parameters(self):

        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index=None, size = None, weights=None, edge_weights = None):

        H, C = self.heads, self.out_channels
        N = x.shape[0]

        # Project node embeddings to feasible dimension, split into multiple heads
        h_prime = x.matmul(weights).view((N, H, C//H))

        # Take dot product between attention vector and all node embeddings,
        # (both for left and right nodes)
        alpha_l = torch.sum(h_prime*self.att_l, dim=-1)
        alpha_r = torch.sum(h_prime*self.att_r, dim=-1)

        if edge_index is None:
            edge_index = torch.combinations(torch.arange(N), r=2).t().contiguous().to(self.device)
            # Propagate embeddings
        out = self.propagate(edge_index=edge_index, x=(h_prime, h_prime), alpha=(alpha_l, alpha_r), size=size, edge_weights=edge_weights)

        # Bring back to original shape
        out = out.view((N, C))

        return out


    def message(self,index, x_j, alpha_j, alpha_i, ptr, size_i, edge_weights):


        # Add left and right dot product
        final_attention_weights = torch.add(alpha_i, alpha_j)
        if edge_weights is None:
            edge_weights = torch.ones(size=(index.size(0),), device=index.device)
        # Pass attention_weights through relu, then multiply each with respective edge weight
        att_unnormalized = torch.mul(F.leaky_relu(final_attention_weights).t(), edge_weights).t()

        # Normalize along neighbour nodes dimension
        att_weights = torch_geometric.utils.softmax(att_unnormalized, index=index, num_nodes=size_i, ptr=ptr, dim=-2)

        # Dropout layer for regularization
        att_weights = torch.nn.functional.dropout(att_weights, p=self.dropout)

        # Multiply embeddings with attention weights
        out = x_j*att_weights.unsqueeze(-1)

        return out


    def aggregate(self, inputs, index, dim_size = None):

        node_dim = self.node_dim

        # Sum all neighbouring edges as output for network
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce="sum")

        return out
    
class GAT(torch.nn.Module):
    def __init__(self, activation, out_features):
        super().__init__()
        self.activation = activation
        attention_dim = 2
        self.w_a = Parameter(torch.Tensor(out_features, attention_dim))
        self.a = Parameter(torch.Tensor(2*attention_dim, 1))

        u.reset_param(self.w_a)
        u.reset_param(self.a)
        self.alpha = 0.001

    def forward(self, node_feats, Ahat, w):
        N = Ahat.shape[0]
        h_prime = node_feats.matmul(w)
        h_reduced = h_prime.matmul(self.w_a)
        H1 = h_reduced.unsqueeze(1).repeat(1,N,1)
        H2 = h_reduced.unsqueeze(0).repeat(N,1,1)
        attn_input = torch.cat([H1, H2], dim = -1) # (N, N, F)
        e = attn_input.matmul(self.a).squeeze(-1) # [N, N]
        attn_mask = -1e18*torch.ones_like(e)
        masked_e = torch.where(Ahat.to_dense() > 0, e, attn_mask)
        attn_scores = F.softmax(masked_e, dim = -1) # [N, N]

        h_prime = torch.mm(attn_scores, h_prime)
        out = self.activation(h_prime)
        return out
    
class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                  args.cols,
                                  torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())

        self.choose_topk = TopK(feats=args.rows,
                                k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        z_topk = self.choose_topk(prev_Z, mask)

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        # the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows, rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows, cols))

    def reset_param(self, t):
        # Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) +
                              self.U.matmul(hidden) +
                              self.bias)

        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)

        self.k = k

    def reset_param(self, t):
        # Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        if mask.shape[-1] != 1:
            mask = mask.view(mask.shape[0], 1)
        scores = scores + mask
        try:
            vals, topk_indices = scores.view(-1).topk(self.k)
        except:
            vals, topk_indices = scores.view(-1).topk(scores.shape[0])

        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = u.pad_with_last_val(topk_indices, self.k)

        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1, 1))

        # we need to transpose the output
        return out.t()
