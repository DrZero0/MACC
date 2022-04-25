# MACC: Multi-Agent Concentrative Coordination with Decentralized Task Representation

This is the implementation of the paper "Multi-Agent Concentrative Coordination with Decentralized Task Representation". This repo is currently maintained by the [LAMDA-RL](https://github.com/LAMDA-RL) group.

Note: the experiments of MAIC is conducted in SC2.4.6.2.69232, which is same as the SMAC run data release (https://github.com/oxwhirl/smac/releases/tag/v1). The results are not always comparable with results run in SC2.4.10.

## Installation instructions
Our version of Python is 3.7.9. 

Set up StarCraft II and SMAC:
```shell
cd pymarl-master
bash install_sc2.sh
```

This will download SC2.4.6.2.69232 into the 3rdparty folder and copy the maps necessary to run over. You may also need to set the environment variable for SC2:

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

Install packages:
```shell
pip install -r requirements.txt
```

Install lb-foraging:
```shell
cd ..
cd MACC_lbforaging
cd lb-foraging
pip install -e .
```

## Run an experiment 

Run an experiment on 5m\_vs\_6m of SMAC: 
```shell
python src/main.py --config=macc --env-config=sc2 with env_args.map_name=5m_vs_6m
```

Run an experiment on Level-Based Foraging (LBF): 
```shell
python src/main.py --config=macc --env-config=foraging
```

Run an experiment on Predator-Prey (PP): 
```shell
python src/main.py --config=macc --env-config=pred_prey_punish
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `results` folder.