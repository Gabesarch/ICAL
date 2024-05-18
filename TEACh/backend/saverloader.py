import os
import pathlib
import torch
import glob
from arguments import args
import ipdb
st = ipdb.set_trace

def save_checkpoint(model, checkpoint_dir, step, epoch, optimizer, keep_latest=3, lr_scheduler=None):
    model_name = "model-%08d.pth"%(epoch)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    prev_chkpts = list(pathlib.Path(checkpoint_dir).glob('model-*'))
    prev_chkpts.sort(key=lambda p: p.stat().st_mtime,reverse=True)
    if len(prev_chkpts) > keep_latest-1:
        for f in prev_chkpts[keep_latest-1:]:
            f.unlink()
    path = os.path.join(checkpoint_dir, model_name)
    if step is None:
        step=0
    if epoch is None:
        epoch=0
    if lr_scheduler is None:
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, path)
    else:
        torch.save({
            'step': step,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict()
            }, path)
    print("Saved a checkpoint: %s"%(path))



def load(model_name, checkpoint_root, model, optimizer, lr_scheduler=None, strict=True):
    print("reading full checkpoint...")
    checkpoint_dir = os.path.join(checkpoint_root, model_name)
    step = 0
    if not os.path.exists(checkpoint_dir):
        print("...ain't no full checkpoint here!")
        print(checkpoint_dir)
        assert(False)
    else:
        ckpt_names = os.listdir(checkpoint_dir)
        steps = [int((i.split('-')[1]).split('.')[0]) for i in ckpt_names]
        if len(ckpt_names) > 0:
            step = max(steps)
            model_name = 'model-%08d.pth' % (step)
            path = os.path.join(checkpoint_dir, model_name)
            print("...found checkpoint %s"%(path))

            checkpoint = torch.load(path)
            
            # # Print model's state_dict
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            # input()

            # # Print optimizer's state_dict
            # print("Optimizer's state_dict:")
            # for var_name in optimizer.state_dict():
            #     print(var_name, "\t", optimizer.state_dict()[var_name])
            # input()
            
            
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if lr_scheduler is not None:
                if 'lr_scheduler_state_dict' in checkpoint.keys():
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                else:
                    "WANRNING: LR SCHEDULER NOT IN CHECKPOINT. Returning lr_scheduler without loading state dict."
        else:
            print("...ain't no full checkpoint here!")
            print(checkpoint_dir)
            assert(False)
    return step


def load_from_path(path, model, optimizer, lr_scheduler=None, strict=True):
    print("reading full checkpoint...")
    # step = 0
    # path = args.load_model_path 

    # step = int((path.split('-')[1]).split('.')[0])
    # print(step)

    # if args.lr_scheduler_from_scratch:
    #     print("LR SCHEDULER FROM SCRATCH")
    #     lr_scheduler_load = False
    # else:
    #     lr_scheduler_load = True

    # if args.optimizer_from_scratch:
    #     print("OPTIMIZER FROM SCRATCH")
    #     optimizer_load = False
    # else:
    #     optimizer_load = True

    checkpoint = torch.load(path)
    # print(checkpoint['step'])
    if 'step' in checkpoint.keys():
        step = int(checkpoint['step'])
        print(f"Start iteration is {step}")
    else:
        step = 0
    if 'epoch' in checkpoint.keys():
        epoch = int(checkpoint['epoch'])
        print(f"Start iteration is {step}")
    else:
        epoch = 0
    # if not strict:
    #     print("REMOVING FRAME_EMBED BEFORE LOADING STATE DICT!!!!!!")
    #     checkpoint['model_state_dict'].pop('model.frame_embed')
        
    #     if args.query_per_action:
    #         print("REMOVING action_embed BEFORE LOADING STATE DICT!!!!!!")
            
    #         for k in list(checkpoint['model_state_dict'].keys()):
    #             if 'model.action_embed' in k or 'model.query_embed' in k:
    #                 checkpoint['model_state_dict'].pop(k)

    #     if args.use_clip_text_encoder:
    #         print("REMOVING text_encoder BEFORE LOADING STATE DICT!!!!!!")
    #         for k in list(checkpoint['model_state_dict'].keys()):
    #             if 'text_encoder' in k or 'resizer' in k:
    #                 checkpoint['model_state_dict'].pop(k)

                    
    # print("REMOVING 'module.' from state dict")
    # checkpoint['model_state_dict'] = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    # print(checkpoint['model_state_dict'].keys())
    module_in_model = False
    for name, param in model.named_parameters():
        if 'module.' in name:
            module_in_model = True
    module_in_checkpoint = False
    for name in checkpoint['model_state_dict'].keys():
        if 'module.' in name:
            module_in_checkpoint = True
    print("module. inmodel?", module_in_model)
    print("module. in checkpoint?", module_in_checkpoint)
    if module_in_model and not module_in_checkpoint:
        print("Adding module. to checkpoint since it is missing...")
        checkpoint['model_state_dict'] = {'module.'+k:v for k, v in checkpoint['model_state_dict'].items()}
    if not module_in_model and module_in_checkpoint:
        print("Removing module. from checkpoint since model does not have it...")
        checkpoint['model_state_dict'] = {k.replace("module.", ""):v for k, v in checkpoint['model_state_dict'].items()}

    if not strict:
        current_model_dict = model.state_dict()
        # adjust for different sized parameters if not strict
        for k, v in checkpoint['model_state_dict'].items():
            if k in current_model_dict.keys():
                if not (checkpoint['model_state_dict'][k].size()==current_model_dict[k].size()):
                    print(f"Size mismatch for param {k}.. setting param in checkpoint to model init (strict=False)")
                    checkpoint['model_state_dict'][k] = current_model_dict[k]
    
    # module_in_model = False
    # for name, param in model.named_parameters():
    #     if 'module.' in name:
    #         module_in_model = True
    # module_in_checkpoint = False
    # for name in checkpoint['model_state_dict'].keys():
    #     if 'module.' in name:
    #         module_in_checkpoint = True
    # print("module_in_model", module_in_model)
    # print("module_in_checkpoint", module_in_checkpoint)
    msg = model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    print(msg)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if lr_scheduler is not None:
        if 'lr_scheduler_state_dict' in checkpoint.keys():
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            "WANRNING: LR SCHEDULER NOT IN CHECKPOINT. Returning lr_scheduler without loading state dict."
    print(f"Loaded {path}")
    return step, epoch