import argparse
import yaml
import os
# assert False
def get_parser(argv=None):
    parser = argparse.ArgumentParser(description='RfS-Net')
    parser.add_argument('--config', type=str, default=None, help='path to config file')
    parser.add_argument('--test_epoch', type=str, default='config/rfs_phase2_scannet.yaml', help='path to config file')
    ### pretrain
    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--run_id', type=str, default=None, help='path to pretrain model')
    parser.add_argument('--resume_with_wandb_run_id', '-r', action='store_true', help='path to pretrain model')
    parser.add_argument('--exp_root', type=str, default='.', help='path to pretrain model')
    parser.add_argument('--bsp_root', type=str, default='datasets/bsp/zs', help='path to pretrain model')
    parser.add_argument('--zs_suffix', type=str, default='_vae1', help='path to pretrain model')
    parser.add_argument('--num_cad_classes', type=int, default=8, help='path to pretrain model')
    parser.add_argument('--apply_max_points_limit', type=bool, default=False, help='path to pretrain model')
    parser.add_argument('--accumulate_grad_batches', type=int, default=None)

    args_cfg = parser.parse_args(argv)
    assert args_cfg.config is not None or args_cfg.run_id is not None
    if os.getenv("NO_WANDB", "0") == "1":
        assert args_cfg.run_id is None

    if args_cfg.config is not None:
        with open(args_cfg.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        for key in config:
            for k, v in config[key].items():
                setattr(args_cfg, k, v)
    else:
        import wandb
        # run_id = f'yuki-kawana-1990/dimr/{args_cfg.run_id}'
        wapi = wandb.Api()

        run = wapi.from_path(args_cfg.run_id)
        cfg_ = run.config
        cfg_ = argparse.Namespace(**{k.split('/')[-1]: v for k, v in cfg_.items()})
        cfg_.run_id = args_cfg.run_id
        cfg_.resume_with_wandb_run_id = args_cfg.resume_with_wandb_run_id
        args_cfg = cfg_

    if not isinstance(args_cfg.accumulate_grad_batches, int) or args_cfg.accumulate_grad_batches is not None:
        args_cfg.accumulate_grad_batches = None


    return args_cfg


cfg = get_parser()
setattr(cfg, 'exp_path', os.path.join(f'{cfg.exp_root}/exp', cfg.dataset, cfg.model_name, cfg.config.split('/')[-1][:-5]))
