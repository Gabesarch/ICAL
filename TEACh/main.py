from arguments import args
import torch
import numpy as np
import random
import threading
import time
import os
import sys

import ipdb
st = ipdb.set_trace

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    print("Mode:", args.mode)
    print(type(args.mode))    
    if 'teach_eval_edh' in args.mode or 'teach_eval_tfd' in args.mode:
        from models.teach_eval_embodied_llm import run_teach
        run_teach()
    elif 'teach_train_depth' in args.mode:
        from models.perception.teach_train_depth import Ai2Thor as Ai2Thor_DEPTH
        aithor_depth = Ai2Thor_DEPTH()
        aithor_depth.run_episodes()
    elif 'teach_skill_learning' in args.mode:
        from models.teach_skill_learning import run_teach
        run_teach()
    elif 'train_inverse_dynamics' in args.mode:
        args.dont_use_controller = True
        from models.train_inverse_dynamics import Ai2Thor as Ai2Thor_IDM
        aithor_idm = Ai2Thor_IDM()
        aithor_idm.run_train()
    else:
        raise NotImplementedError

    print("main finished.")

if __name__ == '__main__':
    main()
