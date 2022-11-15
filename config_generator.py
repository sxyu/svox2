import itertools
from pathlib import Path
import json
from copy import deepcopy

ROOT_PATH = 'opt/configs/auto_tune/fake_sample/'
root_dir = Path(ROOT_PATH)
root_dir.mkdir(parents=True, exist_ok=True)


with (root_dir / 'config.json').open('r') as f:
    tune_conf = json.load(f)
params = tune_conf['params']



ids = []
for i in range(len(params)):
    ids.append(list(range(len(params[i]['values']))))

choices = list(itertools.product(*ids))

source_config = ""

if 'source_conf' in tune_conf:
    with Path(tune_conf['source_conf']).open('r') as f:
        source_config += "########## Source Config ##########\n"
        for line in f:
            if line.startswith('#') or line.startswith('\n'):
                continue
            source_config += line
        source_config += "\n########## Tuned Config ##########\n"

configs = []
check_pairs = [
    ['lr_surface = {}\n', 'lr_surface_final = {}\n']
]
for choice in choices:
    config_record = {} # used to check whether lr_start is larger than lr_final
    
    config = deepcopy(source_config)
    # config += f"include '{tune_conf['source_conf']}'\n\n"
    for i in range(len(params)):
        v = params[i]['values'][choice[i]]
        config_record[params[i]['text']] = v
        if isinstance(v, list):
            config += params[i]['text'].format(*v) + "\n"
        else:
            config += params[i]['text'].format(v) + "\n"


    skip = False
    for pair in check_pairs:
        if pair[0] in config_record and pair[1] in config_record:
            if config_record[pair[0]] < config_record[pair[1]]:
                skip = True
                break
    if skip:
        continue
    configs.append(config)

for i in range(len(configs)):
    filepath = root_dir / f"{i:03d}.yaml"
    with filepath.open("w") as f:
        f.write(configs[i])

    print(filepath)








