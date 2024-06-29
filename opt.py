import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', type=str, default='zwc_exp',
                        help='exp_name')

    # dataset
    parser.add_argument('--root_dir', type=str, default="/data/zwc/nerf_study/data/nerf_360/bicycle",
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='Colmap',
                        help='dataset type, now only Colmap')
    parser.add_argument('--downsample', type=int, default=8,
                        help='downsample of the image, can be 1 2 4 8')
                        
    # train
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='ckp path')
    parser.add_argument('--val_only', default=False, action="store_true",
                        help='val_only')
    parser.add_argument('--num_steps', type=int, default=30000,
                        help='num of num_steps')
    parser.add_argument('--val_check_interval', type=int, default=3000,
                        help='val_check_interval')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='num of gpus')
    parser.add_argument('--no_save_test', default=False, action="store_true",
                        help='no_save_test')

    # model
    parser.add_argument('--white_background', default=False, action='store_true',
                        help='Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.')
    parser.add_argument('--sh_degree', type=int, default=3,
                        help='Order of spherical harmonics to be used (no larger than 3)')
    parser.add_argument('--convert_SHs_python', default=False, action='store_true',
                        help='Flag to make pipeline compute forward and backward of SHs with PyTorch instead of ours.')
    parser.add_argument('--compute_cov3D_python', default=False, action='store_true',
                        help='Flag to make pipeline compute forward and backward of the 3D covariance with PyTorch instead of ours.')
    parser.add_argument('--debug', default=False, action='store_true',
                        help='Enables debug mode if you experience erros. If the rasterizer fails, a dump file is created that you may forward to us in an issue so we can take a look.')
    parser.add_argument('--debug_from', type=int, default=0,
                        help='Debugging is slow. You may specify an iteration (starting from 0) after which the above debugging becomes active.')
    
    # optim
    parser.add_argument('--feature_lr', type=float, default=0.0025,
                        help='Spherical harmonics features learning rate, 0.0025 by default.')
    parser.add_argument('--opacity_lr', type=float, default=0.05,
                        help='Opacity learning rate, 0.05 by default.')
    parser.add_argument('--scaling_lr', type=float, default=0.005,
                        help='Scaling learning rate, 0.005 by default.')
    parser.add_argument('--rotation_lr', type=float, default=0.001,
                        help='Rotation learning rate, 0.001 by default.')
    parser.add_argument('--position_lr_max_steps', type=int, default=30000,
                        help='Number of steps (from 0) where position learning rate goes from initial to final. 30_000 by default.')
    parser.add_argument('--position_lr_init', type=float, default=0.00016,
                        help='Initial 3D position learning rate, 0.00016 by default.')
    parser.add_argument('--position_lr_final', type=float, default=0.0000016,
                        help='Final 3D position learning rate, 0.0000016 by default.')
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01,
                        help='Position learning rate multiplier (cf. Plenoxels), 0.01 by default.')
    parser.add_argument('--densify_from_iter', type=int, default=500,
                        help='Iteration where densification starts, 500 by default.')
    parser.add_argument('--densify_until_iter', type=int, default=15000,
                        help='Iteration where densification stops, 15_000 by default.')
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002,
                        help='Limit that decides if points should be densified based on 2D position gradient, 0.0002 by default.')
    parser.add_argument('--densification_interval', type=int, default=100,
                        help='How frequently to densify, 100 (every 100 iterations) by default.')
    parser.add_argument('--opacity_reset_interval', type=int, default=3000,
                        help='How frequently to reset opacity, 3_000 by default.')
    parser.add_argument('--lambda_dssim', type=float, default=0.2,
                        help='Influence of SSIM on total loss from 0 to 1, 0.2 by default.')
    parser.add_argument('--percent_dense', type=float, default=0.01,
                        help='Percentage of scene extent (0--1) a point must exceed to be forcibly densified, 0.01 by default.')

    return parser.parse_args()