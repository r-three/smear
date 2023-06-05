import torch
import ipdb

class Scorer(object):

    def __init__(self, config):
        self.config = config

        self.ttl_correct = 0
        self.ttl = 0
        if config.dataset == 'DomainNet':
            self.domain_ttl_correct = dict([('domain'+str(i),0) for i in range(self.config.num_domains)])
            self.domain_ttl = dict([('domain'+str(i),0) for i in range(self.config.num_domains)])

    def add_batch(self, batch_idx, batch_pred_lbl, batch_true_lbl=None, domain_lbl=None):
        '''
        Keeps track of the accuracy of current batch

        :param list_idx:
        :param list_:
        :return:
        '''
        if self.config.dataset == 'Shapes':
            self.ttl += batch_pred_lbl.shape[0]
            self.ttl_correct += torch.sum(batch_pred_lbl)
        else:
            crct_idx = batch_pred_lbl == batch_true_lbl

            self.ttl += batch_true_lbl.shape[0]
            self.ttl_correct += torch.sum(crct_idx)

            for i in range(len(domain_lbl)):
                self.domain_ttl['domain'+str(domain_lbl[i].item())] += 1
                self.domain_ttl_correct['domain'+str(domain_lbl[i].item())] += (crct_idx[i]).item()

    def get_score(self):
        '''
        Gets the accuracy
        :return: rounded accuracy to 3 decimal places
        '''

        dict_scores = {}
        round_tot_acc = round(self.ttl_correct.item() / self.ttl, 4)
        dict_scores["acc"] = round_tot_acc
        if self.config.dataset == 'DomainNet':
            for key in self.domain_ttl_correct:
                if self.domain_ttl_correct[key] != 0:
                    dict_scores[key] = round(self.domain_ttl_correct[key] / self.domain_ttl[key], 4)
                else:
                    dict_scores[key] = 0

        return round_tot_acc, dict_scores