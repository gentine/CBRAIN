#-*- coding: utf-8 -*-
import argparse
import configargparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = configargparse.ArgParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
#net_arg.add_argument('--input_scale_size', type=int, default=64, help='input image will be resized with the given value as width and height')
#net_arg.add_argument('--conv_hidden_num', type=int, default=128, choices=[64, 128,16,32],help='n in the paper')
net_arg.add_argument('--hidden',  type=str, default='1024,1024', help='comma separated list of hidden layer units')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--output_names', type=str, default='SPDT,SPDQ', help='names of predicted variable(s)')
data_arg.add_argument('--batch_size', type=int, default=256)
data_arg.add_argument('--frac_train', type=float, default=0.8)
data_arg.add_argument('--local', type=str2bool, default=False)
data_arg.add_argument('--epoch', type=int, default=200)
data_arg.add_argument('--randomize', type=str2bool, default=True)
data_arg.add_argument('--convo', type=str2bool, default=False)
data_arg.add_argument('--localConvo', type=str2bool, default=False)
data_arg.add_argument('--filter_size', type=int, default=3)
data_arg.add_argument('--input_names', type=str, default="TBP,QBP,PS,lat,SOLIN,SHFLX,LHFLX,dTdt_adiabatic,dQdt_adiabatic", help='names of input variables')
data_arg.add_argument('--normalizeInoutputs', type=str2bool, default=True, help='noarmalize data using mean and std nc files')
data_arg.add_argument('--convert_units', type=str2bool, default=False, help='flag to convert SPDQ and Q tendencies into T tendencies')
data_arg.add_argument('--nlevs_imposed', type=int, default=0, help='only use 1 to nlevs_imposed (for instance not stratosphere)')


# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=1000000)
train_arg.add_argument('--lr_update_step', type=int, default=1000)
train_arg.add_argument('--lr_update_epoch', type=int, default=1)
train_arg.add_argument('--lr', type=float, default=0.001)
train_arg.add_argument('--lr_lower_boundary', type=float, default=2e-8)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)
train_arg.add_argument('--gamma', type=float, default=0.5)
train_arg.add_argument('--lambda_k', type=float, default=0.001)
train_arg.add_argument('--use_gpu', type=str2bool, default=True)
train_arg.add_argument('--run_validation', type=str2bool, default=True)
train_arg.add_argument('--keep_dropout_rate', type=float, default=1.)
train_arg.add_argument('--trivial_init', type=int, default=0)
train_arg.add_argument('--lossfct', type=str, default="mse") #abs, mse, logloss, Rsquared

# Misc
#parser.add('-c', '--config', default='', is_config_file=True, help='config file path')
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--save_step', type=int, default=5000)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--random_seed', type=int, default=123)
misc_arg.add_argument('--act', type=str, default='relu')
misc_arg.add_argument('--addon', type=str, default='')

def get_config():
    config, unparsed = parser.parse_known_args()
    if config.use_gpu:
        data_format = 'NCHW'
    else:
        data_format = 'NHWC'
    setattr(config, 'data_format', data_format)
    return config, unparsed
