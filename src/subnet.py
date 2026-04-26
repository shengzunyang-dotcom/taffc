import torch
import torch.nn as nn
class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)


class BinaryStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero_index = torch.abs(input) > 1
        middle_index = (torch.abs(input) <= 1) * (torch.abs(input) > 0.4)
        additional = 2 - 4 * torch.abs(input)
        additional[zero_index] = 0.
        additional[middle_index] = 0.4
        return grad_input * additional
        
class MaskedElementWiseVector(nn.Module):
    def __init__(self, vector_size):
        super(MaskedElementWiseVector, self).__init__()
        self.vector_size = vector_size
        self.weight = nn.Parameter(torch.randn(vector_size))
        self.threshold = nn.Parameter(torch.randn(vector_size))
        self.step = BinaryStep.apply
        self.mask = None
        self.keep_ratio = 0
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        with torch.no_grad():
            # std = self.weight.std()
            self.threshold.data.fill_(0.)

    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        #threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight - self.threshold
        mask = self.step(abs_weight)
        # print(mask)
        ratio = torch.sum(mask) / mask.numel()
        self.keep_ratio = ratio
        self.mask = mask
        # print("keep ratio {:.2f}".format(ratio), torch.sum(mask), mask.numel(), mask.grad, self.weight.grad, self.threshold.grad)
        if ratio <= 0.01:
            with torch.no_grad():
                # std = self.weight.std()
                self.threshold.data.fill_(0.)
            abs_weight = torch.abs(self.weight)
            abs_weight = abs_weight - self.threshold
            mask = self.step(abs_weight)
        masked_weight = self.weight * mask
        output = input * masked_weight
        return output, masked_weight
    
class MaskedElementWiseVectorNoise(nn.Module):
    def __init__(self, vector_size):
        super(MaskedElementWiseVectorNoise, self).__init__()
        self.vector_size = vector_size
        self.weight = nn.Parameter(torch.randn(vector_size))
        self.threshold = nn.Parameter(torch.randn(vector_size))
        self.step = BinaryStep.apply
        self.mask = None
        self.keep_ratio = 0

        self.mean = 0.0
        self.stddev = 0.1

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight)
        with torch.no_grad():
            # std = self.weight.std()
            self.threshold.data.fill_(0.)

    def forward(self, input):
        abs_weight = torch.abs(self.weight)
        #threshold = self.threshold.view(abs_weight.shape[0], -1)
        abs_weight = abs_weight - self.threshold
        mask = self.step(abs_weight)
        # print(mask)
        ratio = torch.sum(mask) / mask.numel()
        self.keep_ratio = ratio
        self.mask = mask
        # print("keep ratio {:.2f}".format(ratio), torch.sum(mask), mask.numel(), mask.grad, self.weight.grad, self.threshold.grad)
        if ratio <= 0.01:
            with torch.no_grad():
                # std = self.weight.std()
                self.threshold.data.fill_(0.)
            abs_weight = torch.abs(self.weight)
            abs_weight = abs_weight - self.threshold
            mask = self.step(abs_weight)
        masked_weight = self.weight * mask
        output = input * masked_weight

        noise_device = output.device
        noise = torch.normal(mean=self.mean, std=self.stddev, size=output.size(), device=noise_device)
        output = output + noise * (mask.view(1, 1, -1) == 0)

        return output, masked_weight


class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class VariationalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, device):
        super(VariationalEncoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        self.device = device
        # decoder
        # self.decoder = nn.Sequential(
        #     nn.Linear(2, latent_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(latent_dim, hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Linear(hidden_dim, input_dim),
        #     nn.Sigmoid()
        #     )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)      
        z = mean + var*epsilon
        return z

    # def decode(self, x):
    #     return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        # x_hat = self.decode(z)
        return z