import torch
import torch.nn.functional as F
import torch.nn as nn
import time

class MPNN(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(MPNN, self).__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            #nn.Linear(n_feature, 3*n_feature),
            #nn.ReLU(),
            #nn.Linear(3*n_feature, 1*n_feature),
            #nn.ReLU(),
            nn.Linear(1*n_edge_feature, n_atom_feature*n_atom_feature),
            #nn.ReLU(),
        )
        self.A = nn.Parameter(torch.zeros(size=(n_atom_feature, n_atom_feature)))

    def forward(self, x1, x2, edge):
        message_matrix = self.cal_message(edge)

        message_matrix = message_matrix.view(edge.size(0), edge.size(1), \
                                    edge.size(2), x1.size(-1), x1.size(-1))
        x_repeat = x2.unsqueeze(1).repeat(1,x1.size(1),1,1).unsqueeze(-2)
        
        message = torch.einsum('abcde,abcef->abcdf', (x_repeat, message_matrix))
        message = message.squeeze(-2)
        message = message.sum(2).squeeze()

        reshaped_message = message.view(-1,x1.size(-1))
        reshaped_x = x1.view(-1,x1.size(-1))
        retval = self.C(reshaped_message, reshaped_x)
        retval = retval.view(x1.size(0), x1.size(1), x1.size(2))
        return retval


class EdgeConv(torch.nn.Module):
    def __init__(self, n_edge_feature, n_atom_feature):
        super(EdgeConv, self).__init__()

        self.W = nn.Linear(n_atom_feature, n_atom_feature)
        #self.M = nn.Linear(n_atom_feature, n_atom_feature)
        self.M = nn.Linear(n_edge_feature+n_atom_feature, n_atom_feature)

    def forward(self, x1, x2, edge, valid_edge):
        """
        Dimensions of each variable
        x1 = [, n_mol1_atom, n_property(args.dim_gnn)]
        x2 = [, n_mol2_atom, n_property]
        edge = [, n_mol1_atom, n_mol2_atom, 3]
        valid_edge = [, n_mol1_atom, n_mol2_atom]
        new_edge = [, n_mol1_atom, n_mol2_atom, n_property + 3(x,y,z)]
        m1 = [, n_mol1_atom, n_property]
        m2 = [, n_mol1_atom, n_property]
        retval = [, n_mol1_atom, n_property]

        1. new_edge -> concat information between mol2 and edge vector
        2. m1 -> embed the mol1's information
        3. m2 -> maximum values of certain properties among elements of mol2 at each element in mol1
        4. relu(m1 + m2) -> return value and this becomes next molecule's information(new_h1 | new_h2)

        :param x1: atom feature matrix one-hot vector of ligand molecule after node embedding
        :param x2: atom feature matrix one-hot vector of protein molecule after node embedding
        :param edge: distance matrix between every atoms of m1 and m2
        :param valid_edge: distance(sum of square of distance matrix) between each molecule's atom
        :return: message-passed molecule information vector same size with X1
        """
        new_edge = torch.cat([x2.unsqueeze(1).repeat(1,x1.size(1),1,1), edge], -1)                      
        retval = 0

        m1 = self.W(x1)
        #m2 = (self.M(new_edge)*valid_edge.unsqueeze(-1).\
        #                    repeat(1,1,1,m1.size(-1))).max(2)[0]
        m2 = (self.M(new_edge)*
                valid_edge.unsqueeze(-1).repeat(1,1,1,m1.size(-1))).max(2)[0]
        retval = F.relu(m1+m2)

        return retval


class IntraNet(torch.nn.Module):
    def __init__(self, n_atom_feature, n_edge_feature):
        super(IntraNet, self).__init__()

        self.C = nn.GRUCell(n_atom_feature, n_atom_feature)
        self.cal_message = nn.Sequential(
            nn.Linear(n_atom_feature*2+n_edge_feature, n_atom_feature),
            nn.ReLU(),
            nn.Linear(n_atom_feature, n_atom_feature*3),
            nn.ReLU(),
            nn.Linear(n_atom_feature*3, n_atom_feature),
        )

    def forward(self, edge, adj, x):
        h1 = x.unsqueeze(1).repeat(1,x.size(1),1,1)
        h2 = x.unsqueeze(2).repeat(1,1,x.size(1),1)

        concat = torch.cat([h1,h2,edge],-1)
        message = self.cal_message(concat)
        message = message*adj.unsqueeze(-1).repeat(1,1,1, message.size(-1))
        message = message.sum(2).squeeze()

        #norm = torch.norm(message, p=2, dim=-1, keepdim=True)
        #message = message.div(norm.expand_as(message))
        norm = adj.sum(2, keepdim=True)
        message = message.div(norm.expand_as(message)+1e-6)

        reshaped_message = message.view(-1,x.size(-1))
        reshaped_x = x.view(-1,x.size(-1))
        retval = self.C(reshaped_message, reshaped_x)
        retval = retval.view(x.size(0), x.size(1), x.size(2))

        return retval


class GAT_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GAT_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        #self.A = nn.Parameter(torch.Tensor(n_out_feature, n_out_feature))
        self.A = nn.Parameter(torch.zeros(size=(n_out_feature, n_out_feature)))
        self.gate = nn.Linear(n_in_feature + n_out_feature, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, adj):
        """
        -graph attention gate
        h = WX [n_atom * n_out_feature]
        e = WXA * tr(WX) + tr(WXA * tr(WX)) [n_atom * n_atom]
        attention = Softmax(torch.where(adj > 1e-6, e, zero_vec) * adj [n_atom * n_atom]
            => if adjacency element's value bigger than 1e-6(if certain relation exists) then attention value becomes e's element at same location else zero vector's element at same location
        self.gate(x + zero_vec) [n_atom * 1]
        h_prime = relu(attention * h) [n_atom * n_out_feature]
        coeff = Sigmoid(self.gate(x + zero_vec)).repeat(1, 1, X.size(-1)) [n_atom * n_in_feature]
            => working as coefficient indicating importance ratio between X and att_result. Coefficient multiplies same attention values to all elements in single row.
        return coeff * X + (1-coeff) * h_prime
        choose attention component via tr(WXB + tr(WX)) that component at the same place
        in adjacency matrix has bigger value than 1e-6 then apply softmax function to
        attention matrix, multiply adjacency matrix to that then multiply it with WX
        :param x:   atom feature one-hot vector of ligand or protein molecule
        :param adj: adjacency matrix of ligand or protein molecule
        :return:    attention-multiplied matrix
        """
        h = self.W(x)
        e = torch.einsum('ijl,ikl->ijk', (torch.matmul(h,self.A), h))
        e = e + e.permute((0,2,1))
        zero_vec = -9e15*torch.ones_like(e) # to make softmax result value to zero
        attention = torch.where(adj > 1e-6, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        #h_prime = torch.matmul(attention, h)
        attention = attention*adj
        h_prime = F.relu(torch.einsum('aij,ajk->aik',(attention, h)))

        coeff = torch.sigmoid(self.gate(torch.cat([x,h_prime], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h_prime
        return retval


class GConv_gate(torch.nn.Module):
    def __init__(self, n_in_feature, n_out_feature):
        super(GConv_gate, self).__init__()
        self.W = nn.Linear(n_in_feature, n_out_feature)
        self.gate = nn.Linear(n_out_feature*2, 1)
    
    def forward(self, x, adj):
        """
        - graph convolution gate
        gc_result = relu(Adj * (WX))
        alpha = sigmoid(gc_result).repeat(1, 1, X.size(-1))
                working as coefficient indicating importance ratio between X and gc_result
        return alpha * X + (1-alpha) * gc_result
        :param x:   atom feature one-hot vector of ligand or protein molecule
        :param adj: adjacency matrix of ligand or protein molecule
        :return:    graph convolution resulting matrix
        """
        m = self.W(x)
        m = F.relu(torch.einsum('xjk,xkl->xjl', (adj.clone(), m)))
        coeff = torch.sigmoid(self.gate(torch.cat([x,m], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*m

        #x = torch.bmm(adj, x)
        return retval


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))
        
    def forward(self, x1, layer, additional_args=None):
        p = torch.sigmoid(self.p_logit)
        if additional_args is None:
            out = layer(self._concrete_dropout(x1, p))
        else:
            out = layer(x1, *aditional_args)
        
        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        
        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
        
        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)
        
        input_dimensionality = x1[0].numel() # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality
        
        regularization = weights_regularizer + dropout_regularizer
        return out, regularization
        
    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                    - torch.log(1 - p + eps)
                    + torch.log(unif_noise + eps)
                    - torch.log(1 - unif_noise + eps))
        
        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
