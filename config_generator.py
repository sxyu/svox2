import itertools
from pathlib import Path
import json
from copy import deepcopy

ROOT_PATH = 'opt/configs/auto_tune/test/'
root_dir = Path(ROOT_PATH)
root_dir.mkdir(parents=True, exist_ok=True)


with (root_dir / 'config.json').open('r') as f:
    tune_conf = json.load(f)
params = tune_conf['params']


# if 'quick_exp' in tune_conf:
#   params.append({'text': "max_steps = {}\n",
#                 'values': [20000 if tune_conf['quick_exp'] else 100000]
#   })

#   params.append({'text': "EvalConfig.niter_runtime_eval = {}\n",
#                 'values': [2000 if tune_conf['quick_exp'] else 25000]
#   })

# else:
# params.append({'text': "EvalConfig.niter_runtime_eval = {}\n",
#             'values': [25000]
# })


ids = []
for i in range(len(params)):
    ids.append(list(range(len(params[i]['values']))))

choices = list(itertools.product(*ids))

source_config = ""

if 'source_conf' in tune_conf:
    with Path(tune_conf['source_conf']).open('r') as f:
        source_config += "########## Source Config ##########\n\n"
        for line in f:
            if not line.startswith('#'):
                source_config += line
        source_config += "\n\n########## Tuned Config ##########\n\n"

configs = []
for choice in choices:
    config = deepcopy(source_config)
    # config += f"include '{tune_conf['source_conf']}'\n\n"
    for i in range(len(params)):
        v = params[i]['values'][choice[i]]
        if isinstance(v, list):
            config += params[i]['text'].format(*v) + "\n"
        else:
            config += params[i]['text'].format(v) + "\n"
    configs.append(config)

for i in range(len(configs)):
    filepath = root_dir / f"{i:03d}.yaml"
    with filepath.open("w") as f:
        f.write(configs[i])

    print(filepath)








