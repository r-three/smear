import json
import os
import ast

from src.utils.util import make_exp_dir

class Config(object):
    def __init__(self, filename=None, kwargs=None, mkdir=True):
        self.dataset = "DomainNet"
        self.model = "resnet"
        self.pretrained_weight = "resnet18"

        self.batch_size = 128
        self.eval_batch_size = 128
        self.num_batches = 100000

        self.lr = 1e-3
        self.optim = "adam"
        self.scheduler = None
        self.warmup_ratio = 0
        self.weight_decay = 0
        self.grad_clip_norm = 1
        self.grad_accum_factor = 1
        self.weight_path = ""
        
        self.seed = 42
        self.exp_dir = None
        self.eval_every = 1000
        self.log_every=200
        self.eval_test = True
        self.debug = False
        self.build_cache = False
        self.test_mode = False
        self.exp_name = None
        self.base_dir = None
        self.save_model=True

        self.full_finetune = False
        self.train_only_router = False
        self.train_layer = -1
        self.forget_relearn = False
        self.forget_every=5000
        self.freeze_router_after = -1
        self.probe_input_features=False
        self.teacher_force=False
        self.num_count_domain_pred = {}
        self.den_count_domain_pred = {}
        self.cluster_momentum = 0.1
        self.cluster_initscale = 0.01
        self.cluster_distance_metric = "l2"
        self.take_max_router = False
        self.add_layer_norm_before_adapter = False
        self.add_layer_norm_after_adapter = True

        # synthetic experiment hyperparams
        self.latent_dim = 4
        self.categorical_dim = 8
        self.encoder_dropout = 0
        self.decoder_dropout = 0
        self.clip_grad_per_module = False
        self.include_router_dim = 500
        self.include_baseline = True
        self.no_scale=False
        self.save_last_checkpoint=False

        self.train_adapters = True
        self.reduction_factor = 32
        self.num_adapters = 6
        self.num_domains = 6
        self.num_lbl = 345
        self.routing_estimator = None
        self.load_loss_weight = 0
        self.load_loss_accm = 0
        self.supervised_loss_weight = 0
        self.supervised_loss_accm = 0
        self.model_dim = 768
        self.adapter_probs_list = []
        self.baseline_vals_list = []
        self.adapter_samples_list = []
        self.bl_reduction_factor = 8
        self.policy_weight = 0.01
        self.policy_entropy_weight = 5e-4
        self.value_function_weight = 0.01
        self.value_loss_type = "Huber"
        self.skill_lr_ratio = 10
        self.num_routers = 1
        self.device = 'cuda'
        self.weight_init_range = 1e-2
        self.non_linearity = 'swish'
        self.adapter_temp = 10.0
        self.anneal_rate = 1e-4
        self.min_temp = 0.5
        self.down_sample_size = 32
        self.train_size = None
        self.same_qdr_skt = False
        self.same_pnt_rel = False
        self.analyze_model = False
        self.analysis_list = []
        self.same_experts_across_routers = False
        self.bias_in_up_sampler = False
        self.use_load_balancing = False
        self.semi_supervised_ratio = 1.0
        self.jitter_noise = 0.0
        self.all_routers = 16
        self.try_sparsemax = False
        self.create_unbalance = False
        self.no_router_bias = True
        self.router_init_scale = 1e-2
        self.normalize_router_weights = True
        self.same_init_then_branch = -1
        self.epsilon_greedy = 0
        self.expert_dropout = 0
        self.token_dropout = 0
        self.renormalize_adapter_probs = True
        self.eval_time = 0
        self.cosine_router = False

        # git cml
        self.save_for_gitcml = False

        # new experiments
        self.average_domain_adapters = False
        if filename:
            self.__dict__.update(json.load(open(filename)))
        if kwargs:
            self.update_kwargs(kwargs)

        if filename or kwargs:
            self.update_exp_config(mkdir)

    def update_kwargs(self, kwargs):
        for (k, v) in kwargs.items():
            try:
                v = ast.literal_eval(v)
            except ValueError:
                v = v
            setattr(self, k, v)

    def update_exp_config(self, mkdir=True):
        '''
        Updates the config default values based on parameters passed in from config file
        '''
        if self.debug:
            base_dir = os.path.join("exp_out", "debug")
        else:
            base_dir = os.path.join("exp_out", self.dataset, self.model)
            if self.exp_name is not None:
                base_dir = os.path.join(base_dir, self.exp_name)
        self.base_dir = base_dir
        if mkdir:
            self.exp_dir = make_exp_dir(base_dir)

        if self.exp_dir is not None:
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            if not self.test_mode:
                self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))

    def to_json(self):
        '''
        Converts parameter values in config to json
        :return: json
        '''
        return json.dumps(self.__dict__, indent=4, sort_keys=True)

    def save_config(self, filename):
        '''
        Saves the config
        '''
        with open(filename, 'w+') as fout:
            fout.write(self.to_json())
            fout.write('\n')