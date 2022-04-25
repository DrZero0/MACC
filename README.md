# Codes for MACC
## Installation instructions
Our version of Python is 3.7.9. 

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

Install packages:
```shell
pip install -r requirements.txt
```

Install lb-foraging:
```shell
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
