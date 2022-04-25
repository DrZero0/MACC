import os
import datetime
import time

ALG = 'macc'
ENV = 'sc2'
MAP = '5m_vs_6m' # '6h_vs_8z, MMM2, 3s_vs_5z'
REMARK = ''

# visible_devices = ['0', '0', '1', '1']
visible_devices = ['0', '0', '1', '1']

alg_config = '--config=' + ALG
env_config = '--env-config=' + ENV
map_config = 'with env_args.map_name=' + MAP
comment_config = '--remarks=' + REMARK

output_folder = './output'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

if __name__ == '__main__':

    if ENV == 'sc2':
        for i in range(len(visible_devices)):
            device = 'export CUDA_VISIBLE_DEVICES=' + visible_devices[i]
            name = '{}_{}'.format(ALG, REMARK)
            log_name = '{time}-{map}-{alg}.out'.format(
                alg=name, 
                map=MAP, 
                time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            )
            command = 'nohup python src/main.py {} {} {} {} {} > {} 2>&1 &'.format(
                alg_config, 
                env_config, 
                map_config,
                't_max=2100000',
                comment_config, 
                os.path.join(output_folder, log_name)
            )
            os.system(device + ';' + command)
            if i != len(visible_devices) - 1:
                time.sleep(61)
    else:
        for i in range(len(visible_devices)):
            device = 'export CUDA_VISIBLE_DEVICES=' + visible_devices[i]
            name = '{}_{}'.format(ALG, REMARK)
            log_name = '{time}-{env}-{alg}.out'.format(
                alg=name, 
                env=ENV, 
                time=datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
            )
            command = 'nohup python src/main.py {} {} {} > {} 2>&1 &'.format(
                alg_config,
                env_config,
                comment_config,  
                os.path.join(output_folder, log_name)
            )
            os.system(device + ';' + command)
            if i != len(visible_devices) - 1:
                time.sleep(61)