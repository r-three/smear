"""model.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import ipdb
from torchvision import transforms
import imageio
import numpy as np
import storch
from src.data.Batcher import Batcher
from src.data.DatasetReader import DatasetReader

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class WAE_Shapes(nn.Module):
    """Encoder-Decoder architecture for both WAE_Shapes-MMD"""
    def __init__(self, config, device):
        super(WAE_Shapes, self).__init__()

        self.device = device
        self.config = config
        self.latent_dim = self.config.latent_dim
        self.categorical_dim = self.config.categorical_dim
        z_dim = self.latent_dim * self.categorical_dim
        if config.include_router_dim != 0:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
                nn.BatchNorm2d(128),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
                nn.BatchNorm2d(256),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
                nn.BatchNorm2d(512),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
                nn.BatchNorm2d(1024),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                View((-1, 1024*4*4)),                                 # B, 1024*4*4
                nn.Linear(1024*4*4, config.include_router_dim) ,                          # B, z_dim
            )
            self.multi_router_batch_norms = []
            self.multi_router = []
            for i in range(self.latent_dim):
                self.multi_router_batch_norms.append(nn.BatchNorm1d(config.include_router_dim))
                self.multi_router.append(nn.Linear(config.include_router_dim, self.categorical_dim))
            self.multi_router_batch_norms = nn.ModuleList(self.multi_router_batch_norms)
            self.multi_router = nn.ModuleList(self.multi_router)
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 128, 4, 2, 1, bias=False),              # B,  128, 32, 32
                nn.BatchNorm2d(128),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),             # B,  256, 16, 16
                nn.BatchNorm2d(256),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),             # B,  512,  8,  8
                nn.BatchNorm2d(512),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                nn.Conv2d(512, 1024, 4, 2, 1, bias=False),            # B, 1024,  4,  4
                nn.BatchNorm2d(1024),
                nn.Dropout(p=self.config.encoder_dropout),
                nn.ReLU(True),
                View((-1, 1024*4*4)),                                 # B, 1024*4*4
                nn.Linear(1024*4*4, 500) ,                          # B, z_dim
                nn.BatchNorm1d(500),
                nn.Linear(500,z_dim)
            )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024*8*8),                           # B, 1024*8*8
            nn.Dropout(p=self.config.decoder_dropout), 
            View((-1, 1024, 8, 8)),                               # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),   # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.Dropout(p=self.config.decoder_dropout), 
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),    # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.Dropout(p=self.config.decoder_dropout), 
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),    # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, 1),                       # B,   nc, 64, 64
        )
        self.weight_init()
        self.mse_loss = nn.MSELoss(reduction='none')
        self.huber_loss = nn.HuberLoss(reduction='none')
        self.relu = nn.ReLU()
        if self.config.supervised_loss_weight != 0:
            self.supervised_loss_fn = nn.CrossEntropyLoss()
        if self.config.routing_estimator == 'reinf_bl_routing':
            if config.include_router_dim != 0:
                self.cv_net = nn.ModuleList([nn.Sequential(
                    nn.Linear(config.include_router_dim,config.include_router_dim//config.bl_reduction_factor),
                    nn.Linear(config.include_router_dim//config.bl_reduction_factor,1)
                ) for _ in range(self.latent_dim)])
            for latent_index in range(self.latent_dim):
                self.cv_net[latent_index][1].weight.data.fill_(0)
                self.cv_net[latent_index][1].bias.data.fill_(0)
                
    def control_variate(self,z):
        return self.cv_net(z)

    def weight_init(self):
        def kaiming_init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        for block in self._modules:
            if len(self._modules[block]) > 0:
                for m in self._modules[block]:
                    kaiming_init(m)
            else:
                kaiming_init(block)

    def gumbel_softmax(self, logits, temperature, hard, predict=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        U = torch.rand(logits.shape).to(self.device)
        if predict:
            y = logits
        else:
            y = logits + (-torch.log(-torch.log(U + 1e-20) + 1e-20))
        y = F.softmax(y / temperature, dim=-1)

        if not hard:
            return y
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).reshape(-1, shape[-1])
        y_hard.scatter_(1, ind.unsqueeze(1), 1)
        y_hard = y_hard.reshape(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        # y_hard = Variable(y_hard -y) + y
        y_hard = y_hard - y.detach() + y
        return y_hard 

    def get_loss(self, x, latents, predict=False):
        batch_size = x.shape[0]
        if self.config.include_router_dim != 0:
            #(Batchsize, input_router_dim)
            encoder_output = self.encoder(x)
            # new_encoder_output = self.router_batch_norm(encoder_output)
            # logits_y_given_x = self.router(new_encoder_output).reshape(batch_size, self.latent_dim, self.categorical_dim)
            new_encoder_output = []
            logits_y_given_x = []
            for i in range(self.latent_dim):
                router_input = self.multi_router_batch_norms[i](encoder_output)
                new_encoder_output.append(router_input.unsqueeze(1))
                logits_y_given_x.append(self.multi_router[i](router_input).unsqueeze(1))
            # (Batchsize, latent dim, categorical dim)
            logits_y_given_x = torch.cat(logits_y_given_x, dim=1)
            # (batchsize, latentdim, router_dim)
            new_encoder_output = torch.cat(new_encoder_output, dim=1)
        else:
            #(BatchSize, latentdim , categorical dim)
            logits_y_given_x = self.encoder(x).reshape(batch_size, self.latent_dim, self.categorical_dim)
            new_encoder_output = logits_y_given_x
        #(BatchSize, latentdim , categorical dim)
        q_y_given_x = F.softmax(logits_y_given_x, dim=-1)
        if predict:
            val, ind = q_y_given_x.reshape(-1, self.categorical_dim).max(dim=-1)
            y = torch.zeros_like(q_y_given_x.reshape(-1, self.categorical_dim))
            if self.config.routing_estimator == 'switch_routing' and (not self.config.no_scale):
                x_indices = torch.arange(ind.shape[0]).to(self.device)
                y[x_indices, ind] = val
            else:
                y.scatter_(1, ind.unsqueeze(1), 1)
            # (Batchsize, latent dim, categorical dim)
            y = y.reshape(batch_size, self.latent_dim, self.categorical_dim)
        else:
            if self.config.routing_estimator == 'switch_routing':
                if self.training and self.config.jitter_noise > 0:
                    r1 = 1 - self.config.jitter_noise
                    r2 = 1 + self.config.jitter_noise
                    noise = (r1 - r2) * torch.rand(q_y_given_x.shape).to(self.device) + r2
                    q_y_given_x_noised = q_y_given_x * noise
                else:
                    q_y_given_x_noised = q_y_given_x
                if self.config.no_scale:
                    with torch.no_grad():
                        val, ind = q_y_given_x.reshape(-1, self.categorical_dim).max(dim=-1)
                    mask1 = torch.zeros_like(q_y_given_x.reshape(-1, self.categorical_dim))
                    mask1.scatter_(1, ind.unsqueeze(1), 1)
                    mask2 = torch.ones_like(q_y_given_x.reshape(-1, self.categorical_dim))
                    x_indices = torch.arange(ind.shape[0]).to(self.device)
                    mask2[x_indices, ind] = val
                    y = (q_y_given_x.reshape(-1, self.categorical_dim) * mask1 ) / mask2
                    y = y.reshape(batch_size, self.latent_dim, self.categorical_dim)
                else:
                    val, ind = q_y_given_x_noised.reshape(-1, self.categorical_dim).max(dim=-1)
                    y = torch.zeros_like(q_y_given_x.reshape(-1, self.categorical_dim))
                    x_indices = torch.arange(ind.shape[0]).to(self.device)
                    y[x_indices, ind] = val
                    y = y.reshape(batch_size, self.latent_dim, self.categorical_dim)

            elif self.config.routing_estimator == 'gs_st_routing':
                #(BatchSize, latent dim, categorical dim)
                y = self.gumbel_softmax(logits_y_given_x.reshape(-1, self.categorical_dim), self.config.adapter_temp, True).reshape(-1, self.latent_dim, self.categorical_dim)

            elif self.config.routing_estimator == 'reinf_bl_routing':
                #(BatchSize, latent dim, categorical dim)
                y_dist = torch.distributions.categorical.Categorical(logits = logits_y_given_x.reshape(-1, self.categorical_dim))
                y = y_dist.sample()
                y = F.one_hot(y, num_classes=self.categorical_dim).reshape(batch_size, self.latent_dim, self.categorical_dim).float()
            else:
                pass
        
        #(BatchSize, 3, 32, 32)
        x_recons = self.decoder(y.reshape(batch_size,-1))
        recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
        return y, logits_y_given_x, new_encoder_output, q_y_given_x, recons_loss

    def forward(self,batch):
        x = batch['input']['image'].to(self.device)
        latents = batch['output']['latent'].to(self.device).float()
        modified_latents = batch['output']['modified_latent'].to(self.device)
        monolithic_latents = batch['output']['monolithic_latent'].to(self.device)
        hash_latents = batch['output']['hash_latent'].to(self.device)
        batch_size = x.shape[0]

        if self.config.routing_estimator == 'tag_routing':
            latents = latents.reshape(batch_size, -1)
            x_recons = self.decoder(latents)
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
            recons_loss = torch.mean(recons_loss)
            return recons_loss, {'loss': recons_loss}
    
        if self.config.routing_estimator == 'hash_routing':
            hash_latents = hash_latents.reshape(batch_size, -1)
            x_recons = self.decoder(hash_latents)
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
            recons_loss = torch.mean(recons_loss)
            return recons_loss, {'loss': recons_loss}

        if self.config.routing_estimator == 'monolithic_routing':
            monolithic_latents = monolithic_latents.reshape(batch_size, -1)
            x_recons = self.decoder(monolithic_latents)
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
            recons_loss = torch.mean(recons_loss)
            return recons_loss, {'loss': recons_loss}

        y, logits_y_given_x, new_encoder_output, q_y_given_x, recons_loss = self.get_loss(x, latents)
        load_loss = torch.tensor(0).to(self.device)
        dict_val = {}

        if self.config.use_load_balancing:
            # (latent dim, categorical dim)
            input_f = torch.sum(y, dim = 0)/batch_size
            # (latent dim, categorical dim)
            prob_f = torch.sum(q_y_given_x, dim = 0)/batch_size
            # (latent dim)
            load_loss = torch.sum(input_f*prob_f, dim = -1) * self.categorical_dim
            #(1)
            load_loss = torch.mean(load_loss)

        if self.config.routing_estimator == 'reinf_bl_routing':
            # (B, latent dim, categorical dim)
            log_q_y_given_x = torch.log(q_y_given_x + 1e-20)
            if self.config.include_router_dim !=0:
                if self.config.include_baseline:
                    #(B, latent dim, categorical dim)
                    entropy_loss = - (log_q_y_given_x) * q_y_given_x
                    entropy_loss = torch.mean(torch.sum(entropy_loss, dim=2))
                    # (B, latent dim)
                    baseline_vals = []
                    for latent_index in range(self.latent_dim):
                        baseline_vals.append(self.cv_net[latent_index](new_encoder_output[:,latent_index].detach()))
                    baseline_vals = torch.cat(baseline_vals, dim=1)
                    # (B, latent dim)
                    batched_loss = recons_loss.unsqueeze(1).repeat(1, self.latent_dim)
                    # (B, latent dim)
                    advantage_vals = batched_loss - baseline_vals
                    # (B, latent dim)
                    policy_loss = torch.sum(log_q_y_given_x*y, dim=-1) * advantage_vals.detach()
                    policy_loss = torch.mean(policy_loss)
                    value_loss = torch.mean(self.huber_loss(batched_loss.detach(), baseline_vals))
                    recons_loss = torch.mean(recons_loss)
                    dict_val['recons_loss'] = recons_loss
                    loss = recons_loss + self.config.policy_weight * policy_loss + \
                        self.config.policy_entropy_weight * entropy_loss + \
                            self.config.value_function_weight * value_loss + \
                                self.config.load_loss_weight * load_loss 
                    dict_val.update({"recons_loss": recons_loss, "load_loss": self.config.load_loss_weight * load_loss,
                        "entropy_loss": self.config.policy_entropy_weight * entropy_loss, "policy_loss": self.config.policy_weight * policy_loss,\
                            "value_loss": self.config.value_function_weight * value_loss})

                else:
                    with torch.no_grad():
                        # (B,)
                        batched_loss = recons_loss.clone()
                    # (B)
                    logq = torch.mean(torch.sum(y*log_q_y_given_x,-1),-1)
                    # (B)
                    policy_loss = logq * batched_loss
                    policy_loss = torch.mean(policy_loss)
                    recons_loss = torch.mean(recons_loss)
                    loss = recons_loss + policy_loss
                    dict_val['recons_loss'] = recons_loss
                    dict_val['policy_loss'] = policy_loss

            else:
                with torch.no_grad():
                    # (B,)
                    batched_loss = recons_loss.clone()
                # (B)
                logq = torch.mean(torch.sum(y*log_q_y_given_x,-1),-1)
                # (B)
                policy_loss = logq * batched_loss
                policy_loss = torch.mean(policy_loss)
                recons_loss = torch.mean(recons_loss)
                loss = recons_loss + policy_loss
                dict_val['recons_loss'] = recons_loss
                dict_val['policy_loss'] = policy_loss
        else:
            recons_loss = torch.mean(recons_loss)
            dict_val = {"recons_loss": recons_loss, "load_loss": self.config.load_loss_weight * load_loss}
            loss = recons_loss + self.config.load_loss_weight * load_loss
        if self.config.supervised_loss_weight != 0:
            supervised_loss = torch.tensor(0.0).to(self.config.device)
            for latent_index in range(self.latent_dim):
                supervised_loss += self.supervised_loss_fn(logits_y_given_x[:,latent_index], modified_latents[:,latent_index])
            dict_val["supervised_loss"]= self.config.supervised_loss_weight * supervised_loss
            loss = loss + self.config.supervised_loss_weight * supervised_loss
        dict_val['loss'] = loss
        return loss, dict_val
    
    def analyze_image(self, index, x, x_recons):
        x_np = x[index].cpu().numpy()
        x_recons_np = x_recons[index].cpu().numpy()
        x_np = x_np * 255
        x_recons_np = np.minimum(x_recons_np*255, np.ones_like(x_recons_np)*255)
        x_recons_np = np.maximum(x_recons_np, np.zeros_like(x_recons_np)*255)
        x_np = x_np.astype(np.uint8)
        x_recons_np = x_recons_np.astype(np.uint8)
        imageio.imsave(self.config.exp_dir + '/debug_image_'+str(index)+'.png', x_np.T)
        imageio.imsave(self.config.exp_dir + '/debug_image_recons_' +str(index)+'.png', x_recons_np.T)   
        return 
    
    def predict(self, batch):
        x = batch['input']['image'].to(self.device)
        latents = batch['output']['latent'].to(self.device).float()
        monolithic_latents = batch['output']['monolithic_latent'].to(self.device)
        hash_latents = batch['output']['hash_latent'].to(self.device)
        batch_size = x.shape[0]
        if self.config.routing_estimator == 'tag_routing':
            x_recons = self.decoder(latents.reshape(batch_size, -1))
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
        elif self.config.routing_estimator == 'hash_routing':
            x_recons = self.decoder(hash_latents.reshape(batch_size, -1))
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
        elif self.config.routing_estimator == 'monolithic_routing':
            monolithic_latents = monolithic_latents.reshape(batch_size, -1)
            x_recons = self.decoder(monolithic_latents)
            recons_loss = torch.sum(self.mse_loss(x, x_recons), dim=[1,2,3])
        else:
            y, logits_y_given_x, new_encoder_output, q_y_given_x, recons_loss = self.get_loss(x, latents, True)

        if self.config.analyze_model:
            # need to use expert_index to accumulate values for analysis
            if self.config.routing_estimator == 'tag_routing':
                y = latents
            elif self.config.routing_estimator == 'hash_routing':
                y = hash_latents
            self.config.analysis_list.append(torch.max(y, dim=-1)[1].permute(1,0))
            self.config.analysis_list.append(torch.max(latents, dim=-1)[1].permute(1,0))
        return -recons_loss 




