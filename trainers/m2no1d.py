from models import M2NO1d
from .base import BaseTrainer


class M2NO1DTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(model_name=args['model_name'], device=args['device'], epochs=args['epochs'], 
                         eval_freq=args['eval_freq'], patience=args['patience'], verbose=args['verbose'], 
                         wandb_log=args['wandb'], logger=args['log'], saving_best=args['saving_best'], 
                         saving_checkpoint=args['saving_checkpoint'], saving_path=args['saving_path'])
    
    def build_model(self, args, **kwargs):
        initializer = self.get_initializer(args['initializer'])
        model = M2NO1d(
            in_channels=args['in_channels'],
            out_channels=args['out_channels'],
            k=args['k'],
            c=args['c'],
            num_layer=args['num_layer'],
            grid_levels=args['grid_levels'],
            resolution=args['sample_resolution'][0],
            base=args['base'],
            bias=args['bias'],
            padding_mode=args['padding_mode'],
            initializer=initializer,
            norm=args['norm'],
            activation=args['activation']
            )
        return model
