import os
import math
import random
import torch
import torch.nn as nn
import re
import numpy as np
import ipdb
from src.resnet.modeling_resnet import resnet18, resnet50
import torch.nn.functional as F

class ResNet_Classifier(torch.nn.Module):
    def __init__(self, config, dataset_reader, device):
        '''

        :param device:
        '''
        super(ResNet_Classifier, self).__init__()
        self.config = config
        self.dataset_reader = dataset_reader
        self.device = device
        
        self.num_lbl = self.dataset_reader.get_num_lbl()
        self.config.num_lbl = self.num_lbl
        if self.config.pretrained_weight == 'resnet50':
            self.model = resnet50(pretrained=True, config=self.config)
        elif self.config.pretrained_weight == 'resnet18':
            self.model = resnet18(pretrained=True, config=self.config)
        self.loss = nn.CrossEntropyLoss(reduction="none")
        self.huber_loss = nn.HuberLoss(reduction='none')
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def get_loss(self, batch):
        '''
        Get classification loss

        :param batch:
        :return:
        '''
        # (Batchsize, 3, 224, 224)
        images = batch['input']['image'].to(self.device)
        # (Batchsize)
        labels = batch['output']['lbl'].to(self.device)
        domain_lbls = batch['output']['domain_lbl'].to(self.device)
        hash_lbls = batch['output']['hash_lbl'].to(self.device)
        orig_domain_lbls = batch['output']['orig_domain_lbl'].to(self.device)
        batch_size = images.shape[0]
        # (BatchSize, num_lbl)
        label_logits = self.model(images, domain_lbls, orig_domain_lbls, hash_lbls)
        label_loss = self.loss(label_logits, labels)

        return label_loss, label_logits

    def forward(self, batch):
        '''
        Compute loss

        :param batch:
        '''
        if self.config.train_adapters:
            self.config.load_loss_accm = torch.tensor(0.0).to(self.device)
            self.config.supervised_loss_accm = torch.tensor(0.0).to(self.device)
            if self.config.routing_estimator == "reinf_bl_routing":
                self.config.adapter_probs_list = []
                self.config.baseline_vals_list = []
                self.config.adapter_samples_list = []
            
        label_loss, label_logits = self.get_loss(batch)
        if self.config.probe_input_features:
            loss = torch.tensor(0).to(self.device)
        else:
            loss = torch.mean(label_loss)
        if self.training and self.config.train_adapters:
            if self.config.routing_estimator == 'reinf_bl_routing':
                #bs, n_enc_layers, 6
                adapter_probs = torch.cat([probs.unsqueeze(1) for probs in self.config.adapter_probs_list], dim=1)
                #bs, n_enc_layers, 6
                samples = torch.cat([sample.unsqueeze(1) for sample in self.config.adapter_samples_list], dim=1)
                #bs, n_enc_layers
                baseline_vals = torch.cat([baseline_val for baseline_val in self.config.baseline_vals_list], dim=1)
                #bs, n_enc_layers
                batched_loss = label_loss.unsqueeze(1).repeat(1, len(self.config.baseline_vals_list))
                log_adapter_probs = torch.log(adapter_probs + 1e-20)
                advantage_vals = batched_loss - baseline_vals
                #bs, n_enc_layers, num_adapters
                entropy_loss = - (log_adapter_probs)*adapter_probs
                entropy_loss = torch.mean(torch.sum(entropy_loss, dim=2))
                #bs, n_enc_layers
                policy_loss = torch.sum(log_adapter_probs*samples, dim=-1) * advantage_vals.detach()
                # need to use mean in order to have stable loss
                policy_loss = torch.mean(policy_loss)
                value_loss = torch.mean(self.huber_loss(batched_loss.detach(), baseline_vals))
                dict_val = {}
                dict_val['label_loss'] = loss
                loss = loss + self.config.policy_weight * policy_loss + \
                    self.config.policy_entropy_weight * entropy_loss + \
                        self.config.value_function_weight * value_loss + \
                            self.config.load_loss_weight * self.config.load_loss_accm + \
                                self.config.supervised_loss_weight * self.config.supervised_loss_accm
                dict_val.update({"load_loss": self.config.load_loss_weight * self.config.load_loss_accm, "supervised_loss": self.config.supervised_loss_weight * self.config.supervised_loss_accm,\
                    "entropy_loss": self.config.policy_entropy_weight * entropy_loss, "policy_loss": self.config.policy_weight * policy_loss,\
                        "value_loss": self.config.value_function_weight * value_loss})
                dict_val['loss'] = loss
            
            elif self.config.routing_estimator == "adamix_routing":
                label_loss_2, label_logits_2 = self.get_loss(batch)
                log_label_logits = F.log_softmax(label_logits, dim=1)
                log_label_logits_2 = F.log_softmax(label_logits_2, dim=1)
                consistency_loss = (self.kl_loss(log_label_logits, log_label_logits_2) + self.kl_loss(log_label_logits_2, log_label_logits)) * 0.5
                dict_val = {"label_loss": loss, "load_loss": self.config.load_loss_weight * self.config.load_loss_accm, "supervised_loss": self.config.supervised_loss_weight * self.config.supervised_loss_accm, "consistency_loss": consistency_loss}
                loss = loss + self.config.load_loss_weight * self.config.load_loss_accm + self.config.supervised_loss_weight * self.config.supervised_loss_accm + consistency_loss
                dict_val["loss"] = loss
            else:
                dict_val = {"label_loss": loss, "load_loss": self.config.load_loss_weight * self.config.load_loss_accm, "supervised_loss": self.config.supervised_loss_weight * self.config.supervised_loss_accm}
                loss = loss + self.config.load_loss_weight * self.config.load_loss_accm + self.config.supervised_loss_weight * self.config.supervised_loss_accm
                dict_val['loss'] = loss

        else:
            dict_val = {'loss': loss}
        return loss, dict_val

    def predict(self, batch):
        '''
        Predict the lbl for batch

        :param batch:
        :param pet:
        :return:
        '''
        _, label_logits = self.get_loss(batch)
        lbl_prob = torch.softmax(label_logits, dim = -1)
        return torch.argmax(lbl_prob, dim=-1), lbl_prob
