import torch
import pdb
import numpy as np
import itertools
from fvcore.nn.precise_bn import get_bn_modules
import json

from ..evaluation import lta_metrics as metrics
from ..utils import distributed as du
from ..utils import logging
from ..utils import misc
from ..tasks.video_task import VideoTask

import os
import json
from PIL import Image

logger = logging.get_logger(__name__)

import ipdb
st = ipdb.set_trace

class MultiTaskClassificationTask(VideoTask):
    checkpoint_metric = "val_top1_noun_err"

    def training_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        loss1 = self.loss_fun(preds[0], labels[:, 0])
        loss2 = self.loss_fun(preds[1], labels[:, 1])
        loss = loss1 + loss2
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )

        step_result = {
            "loss": loss,
            "train_loss": loss.item(),
            "train_top1_verb_err": top1_err_verb.item(),
            "train_top5_verb_err": top5_err_verb.item(),
            "train_top1_noun_err": top1_err_noun.item(),
            "train_top5_noun_err": top5_err_noun.item(),
        }

        return step_result

    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx):
        inputs, labels, _, _ = batch
        preds = self.forward(inputs)
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds[0], labels[:, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds[1], labels[:, 1], (1, 5)
        )
        return {
            "val_top1_verb_err": top1_err_verb.item(),
            "val_top5_verb_err": top5_err_verb.item(),
            "val_top1_noun_err": top1_err_noun.item(),
            "val_top5_noun_err": top5_err_noun.item(),
        }

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        inputs, labels, clip_id, _ = batch
        preds = self.forward(inputs)
        return {
            "preds_verb": preds[0],
            "preds_noun": preds[1],
            "labels": labels,
            "clip_ids": clip_id,
        }

    def test_epoch_end(self, outputs):
        preds_verbs = torch.cat([x["preds_verb"] for x in outputs])
        preds_nouns = torch.cat([x["preds_noun"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        clip_ids = [x["clip_ids"] for x in outputs]
        clip_ids = [item for sublist in clip_ids for item in sublist]

        # Gather all labels from distributed processes.
        preds_verbs = torch.cat(du.all_gather([preds_verbs]), dim=0)
        preds_nouns = torch.cat(du.all_gather([preds_nouns]), dim=0)
        labels = torch.cat(du.all_gather([labels]), dim=0)
        clip_ids = list(itertools.chain(*du.all_gather_unaligned(clip_ids)))

        # Ensemble multiple predictions of the same view together. This relies on the
        # fact that the dataloader reads multiple clips of the same video at different
        # spatial crops.
        video_labels = {}
        video_verb_preds = {}
        video_noun_preds = {}
        assert preds_verbs.shape[0] == preds_nouns.shape[0]
        for i in range(preds_verbs.shape[0]):
            vid_id = clip_ids[i]
            video_labels[vid_id] = labels[i]
            if vid_id not in video_verb_preds:
                video_verb_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[0]),
                    device=preds_verbs.device,
                    dtype=preds_verbs.dtype,
                )
                video_noun_preds[vid_id] = torch.zeros(
                    (self.cfg.MODEL.NUM_CLASSES[1]),
                    device=preds_nouns.device,
                    dtype=preds_nouns.dtype,
                )

            if self.cfg.DATA.ENSEMBLE_METHOD == "sum":
                video_verb_preds[vid_id] += preds_verbs[i]
                video_noun_preds[vid_id] += preds_nouns[i]
            elif self.cfg.DATA.ENSEMBLE_METHOD == "max":
                video_verb_preds[vid_id] = torch.max(
                    video_verb_preds[vid_id], preds_verbs[i]
                )
                video_noun_preds[vid_id] = torch.max(
                    video_noun_preds[vid_id], preds_nouns[i]
                )

        video_verb_preds = torch.stack(list(video_verb_preds.values()), dim=0)
        video_noun_preds = torch.stack(list(video_noun_preds.values()), dim=0)
        video_labels = torch.stack(list(video_labels.values()), dim=0)
        top1_verb_err, top5_verb_err = metrics.topk_errors(
            video_verb_preds, video_labels[:, 0], (1, 5)
        )
        top1_noun_err, top5_noun_err = metrics.topk_errors(
            video_noun_preds, video_labels[:, 1], (1, 5)
        )
        errors = {
            "top1_verb_err": top1_verb_err,
            "top5_verb_err": top5_verb_err,
            "top1_noun_err": top1_noun_err,
            "top5_noun_err": top5_noun_err,
        }
        for k, v in errors.items():
            self.log(k, v.item())


class LongTermAnticipationTaskGPT4V(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpoint_metric = f"val_0_ED_{cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT-1}"
        # self.taxonomy = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')
        if os.path.exists(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')):
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')) as json_data:
                self.taxonomy = json.load(json_data)

            self.verbs2idx = {self.taxonomy['verbs'][idx]:idx for idx in range(len(self.taxonomy['verbs']))}
            self.idx2verbs = {v:k for k,v in self.verbs2idx.items()}
            self.nouns2idx = {self.taxonomy['nouns'][idx]:idx for idx in range(len(self.taxonomy['nouns']))}
            self.idx2nouns = {v:k for k,v in self.nouns2idx.items()}

    def forward(self, inputs, tgts):
        return self.model(inputs, tgts=tgts)

    def training_step(self, batch, batch_idx):
        # Labels is tensor of shape (batch_size, time, label_dim)
        input, labels, observed_labels, _, _ = batch

        # Preds is a list of tensors of shape (B, Z, C), where
        # - B is batch size,
        # - Z is number of future predictions,
        # - C is the class
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.forward(input, tgts=labels)
        assert len(preds) == len(self.cfg.MODEL.NUM_CLASSES), len(preds)

        loss = 0
        step_result = {}
        for head_idx, pred_head in enumerate(preds):
            for seq_idx in range(pred_head.shape[1]):

                loss += self.loss_fun(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx]
                )
                top1_err, top5_err = metrics.distributed_topk_errors(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx], (1, 5)
                )

                step_result[f"train_{seq_idx}_{head_idx}_top1_err"] = top1_err.item()
                step_result[f"train_{seq_idx}_{head_idx}_top5_err"] = top5_err.item()

        for head_idx in range(len(preds)):
            step_result[f"train_{head_idx}_top1_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top1" in k]
            )
            step_result[f"train_{head_idx}_top5_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top5" in k]
            )

        step_result["loss"] = loss
        step_result["train_loss"] = loss.item()

        return step_result

    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx, return_images_only=False):
        input, forecast_labels, video_labels, _, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # vid_num = 0
        # std = self.cfg.DATA.STD
        # mean = self.cfg.DATA.MEAN
        # self.image_mean = torch.from_numpy(np.array([mean]).reshape(1,3,1,1)).cpu() #cuda()
        # self.image_std = torch.from_numpy(np.array([std]).reshape(1,3,1,1)).cpu() #cuda()
        # video_image = torch.cat(input[0][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
        # video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        # video_image.save(f'images/test{vid_num}.png')
        # video_image = torch.cat(input[1][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
        # video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        # video_image.save(f'images/test{vid_num}_full.png')

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds_forecast, preds_recog, response_idm, response_ltp, images_samples_ltp, examples_retrieved = self.model.generate(input, k=k, return_images_only=return_images_only)  # [(B, K, Z)]
        
        # inputs, labels, _, _ = batch
        # preds = self.forward(inputs)
        # loss1 = self.loss_fun(preds[0], labels[:, 0])
        # loss2 = self.loss_fun(preds[1], labels[:, 1])
        # loss = loss1 + loss2

        # preds_rec = torch.zeros(1, len(self.verbs2idx), 4)

        top_max_k_inds = preds_recog[0].squeeze(0).t()
        top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
            preds_recog[0].squeeze(0).t(), video_labels[:, :, 0], (1, 5)
        )
        top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
            preds_recog[1].squeeze(0).t(), video_labels[:, :, 1], (1, 5)
        )

        print()
        print()
        print("Video GT labels:")
        for l_idx in range(video_labels.shape[1]):
            print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
        print()
        print()
        print("Forecasting GT labels:")
        for l_idx in range(forecast_labels.shape[1]):
            print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
        print()
        print()

        step_result = {
            # "loss": loss,
            # "train_loss": loss.item(),
            "top1_verb_err": top1_err_verb.item(),
            "top5_verb_err": top5_err_verb.item(),
            "top1_noun_err": top1_err_noun.item(),
            "top5_noun_err": top5_err_noun.item(),
        }
        
        # step_result = {}
        for head_idx, pred in enumerate(preds_forecast):
            assert pred.shape[1] == 5
            bi, ki, zi = (0, 1, 2)
            pred = pred.permute(bi, zi, ki)
            pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

            label = forecast_labels[:, :, head_idx : head_idx + 1]
            auedit = metrics.distributed_AUED(pred, label)
            results = {
                f"val_{head_idx}_" + k: v for k, v in auedit.items()
            }
            step_result.update(results)
        # st()
        # torch.stack(preds_forecast).permute(0,3,2,1).squeeze(-1)
        # auedit = metrics.distributed_AUED(torch.stack(preds_forecast).permute(0,3,2,1).squeeze(-1), forecast_labels.permute(2,1,0))
        auedit = metrics.distributed_AUED(torch.sum(torch.stack(preds_forecast).permute(0,3,2,1).squeeze(-1), dim=0).unsqueeze(0), torch.sum(forecast_labels.permute(2,1,0), dim=0).unsqueeze(0))
        results = {
                f"sum_" + k: v for k, v in auedit.items()
            }
        step_result.update(results)
        step_result["response_idm"] = response_idm
        step_result["response_ltp"] = response_ltp
        step_result["examples_retrieved"] = examples_retrieved
        return step_result, images_samples_ltp

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        input, forecast_labels, video_labels, last_clip_ids, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])

        # print()
        # print()
        # print("Video labels:")
        # for l_idx in range(video_labels.shape[1]):
        #     print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()
        # print("Forecasting labels:")
        # for l_idx in range(forecast_labels.shape[1]):
        #     print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()
        # st()

        preds = self.model.generate(input, k=k)  # [(B, K, Z)]

        compute_metrics = True
        if not torch.all(forecast_labels==-1) and compute_metrics:
            step_result = {}
            for head_idx, pred in enumerate(preds):
                assert pred.shape[1] == k
                bi, ki, zi = (0, 1, 2)
                pred = pred.permute(bi, zi, ki)
                pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

                label = forecast_labels[:, :, head_idx : head_idx + 1]
                auedit = metrics.distributed_AUED(pred, label)
                results = {
                    f"val_{head_idx}_" + k: v for k, v in auedit.items()
                }
                step_result.update(results)
        
        visualize = True
        if visualize:
            vid_num = 0
            std = self.cfg.DATA.STD
            mean = self.cfg.DATA.MEAN
            self.image_mean = torch.from_numpy(np.array([mean]).reshape(1,3,1,1)).cuda()
            self.image_std = torch.from_numpy(np.array([std]).reshape(1,3,1,1)).cuda()
            video_image = torch.cat(input[0][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
            video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
            video_image.save(f'images/test{vid_num}.png')
            video_image = torch.cat(input[1][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
            video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
            video_image.save(f'images/test{vid_num}_full.png')
            print()
            print()
            print("Video labels:")
            for l_idx in range(video_labels.shape[1]):
                print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
            print()
            print()
            print("Forecasting labels:")
            for l_idx in range(forecast_labels.shape[1]):
                print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
            print()
            print()
            st()
        
        return {
            'last_clip_ids': last_clip_ids,
            'verb_preds': preds[0],
            'noun_preds': preds[1],
        }

    def test_epoch_end(self, outputs):

        test_outputs = {}
        for key in ['verb_preds', 'noun_preds']:
            preds = torch.cat([x[key] for x in outputs], 0)
            preds = self.all_gather(preds).unbind()
            test_outputs[key] = torch.cat(preds, 0)

        last_clip_ids = [x['last_clip_ids'] for x in outputs]
        last_clip_ids = [item for sublist in last_clip_ids for item in sublist]
        last_clip_ids = list(itertools.chain(*du.all_gather_unaligned(last_clip_ids)))
        test_outputs['last_clip_ids'] = last_clip_ids

        if du.get_local_rank() == 0:
            pred_dict = {}
            for idx in range(len(test_outputs['last_clip_ids'])):
                pred_dict[test_outputs['last_clip_ids'][idx]] = {
                    'verb': test_outputs['verb_preds'][idx].cpu().tolist(),
                    'noun': test_outputs['noun_preds'][idx].cpu().tolist(),
                }       
            json.dump(pred_dict, open('outputs.json', 'w'))

class LongTermAnticipationTask(VideoTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.checkpoint_metric = f"val_0_ED_{cfg.FORECASTING.NUM_ACTIONS_TO_PREDICT-1}"
        # self.taxonomy = os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')
        if os.path.exists(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')):
            with open(os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR, 'fho_lta_taxonomy.json')) as json_data:
                self.taxonomy = json.load(json_data)

            self.verbs2idx = {self.taxonomy['verbs'][idx]:idx for idx in range(len(self.taxonomy['verbs']))}
            self.idx2verbs = {v:k for k,v in self.verbs2idx.items()}
            self.nouns2idx = {self.taxonomy['nouns'][idx]:idx for idx in range(len(self.taxonomy['nouns']))}
            self.idx2nouns = {v:k for k,v in self.nouns2idx.items()}
        if not hasattr(self.cfg, 'CHECKPOINT_FILE_PATH') or self.cfg.CHECKPOINT_FILE_PATH=='':
            self.cfg.CHECKPOINT_FILE_PATH = "/media/gsarch/HDD14TB/datasets/ego4d/v2/lta_models/lta_slowfast_trf_v2.ckpt"
        
        checkpoint = torch.load(self.cfg.CHECKPOINT_FILE_PATH)
        checkpoint['state_dict'] = {k.replace('model.', ''):v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)

        self.results_dict = {}

    def forward(self, inputs, tgts):
        return self.model(inputs, tgts=tgts)

    def training_step(self, batch, batch_idx):
        # Labels is tensor of shape (batch_size, time, label_dim)
        input, labels, observed_labels, _, _ = batch

        # Preds is a list of tensors of shape (B, Z, C), where
        # - B is batch size,
        # - Z is number of future predictions,
        # - C is the class
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.forward(input, tgts=labels)
        assert len(preds) == len(self.cfg.MODEL.NUM_CLASSES), len(preds)

        loss = 0
        step_result = {}
        for head_idx, pred_head in enumerate(preds):
            for seq_idx in range(pred_head.shape[1]):

                loss += self.loss_fun(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx]
                )
                top1_err, top5_err = metrics.distributed_topk_errors(
                    pred_head[:, seq_idx], labels[:, seq_idx, head_idx], (1, 5)
                )

                step_result[f"train_{seq_idx}_{head_idx}_top1_err"] = top1_err.item()
                step_result[f"train_{seq_idx}_{head_idx}_top5_err"] = top5_err.item()

        for head_idx in range(len(preds)):
            step_result[f"train_{head_idx}_top1_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top1" in k]
            )
            step_result[f"train_{head_idx}_top5_err"] = np.mean(
                [v for k, v in step_result.items() if f"{head_idx}_top5" in k]
            )

        step_result["loss"] = loss
        step_result["train_loss"] = loss.item()

        return step_result

    def training_epoch_end(self, outputs):
        if self.cfg.BN.USE_PRECISE_STATS and len(get_bn_modules(self.model)) > 0:
            misc.calculate_and_update_precise_bn(
                self.train_loader, self.model, self.cfg.BN.NUM_BATCHES_PRECISE
            )
        _ = misc.aggregate_split_bn_stats(self.model)

        keys = [x for x in outputs[0].keys() if x != "loss"]
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def validation_step(self, batch, batch_idx, return_images_only=False):
        input, forecast_labels, video_labels, _, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # vid_num = 0
        # std = self.cfg.DATA.STD
        # mean = self.cfg.DATA.MEAN
        # self.image_mean = torch.from_numpy(np.array([mean]).reshape(1,3,1,1)).cpu() #cuda()
        # self.image_std = torch.from_numpy(np.array([std]).reshape(1,3,1,1)).cpu() #cuda()
        # video_image = torch.cat(input[0][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
        # video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        # video_image.save(f'images/test{vid_num}.png')
        # video_image = torch.cat(input[1][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
        # video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
        # video_image.save(f'images/test{vid_num}_full.png')

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
        preds = self.model.generate(input, k=k)  # [(B, K, Z)]
        # preds_forecast, preds_recog, response_idm, response_ltp, images_samples_ltp = self.model.generate(input, k=k, return_images_only=return_images_only)  # [(B, K, Z)]
        
        # inputs, labels, _, _ = batch
        # preds = self.forward(inputs)
        # loss1 = self.loss_fun(preds[0], labels[:, 0])
        # loss2 = self.loss_fun(preds[1], labels[:, 1])
        # loss = loss1 + loss2

        # preds_rec = torch.zeros(1, len(self.verbs2idx), 4)

        # top_max_k_inds = preds_recog[0].squeeze(0).t()
        # top1_err_verb, top5_err_verb = metrics.distributed_topk_errors(
        #     preds_recog[0].squeeze(0).t(), video_labels[:, :, 0], (1, 5)
        # )
        # top1_err_noun, top5_err_noun = metrics.distributed_topk_errors(
        #     preds_recog[1].squeeze(0).t(), video_labels[:, :, 1], (1, 5)
        # )

        # print()
        # print()
        # print("Video GT labels:")
        # for l_idx in range(video_labels.shape[1]):
        #     print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()




        # print("Forecasting GT labels:")
        # for l_idx in range(forecast_labels.shape[1]):
        #     print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()
        # print("Estimated labels:")
        # idx2 = 0
        # for l_idx in range(forecast_labels.shape[1]):
        #     print(self.idx2verbs[int(preds[0][0,idx2,l_idx].cpu().numpy())], self.idx2nouns[int(preds[1][0,idx2,l_idx].cpu().numpy())])
        # print()
        # print()

        # step_result = {
        #     # "loss": loss,
        #     # "train_loss": loss.item(),
        #     "top1_verb_err": 100.,
        #     "top5_verb_err": top5_err_verb.item(),
        #     "top1_noun_err": top1_err_noun.item(),
        #     "top5_noun_err": top5_err_noun.item(),
        # }
        
        # step_result = {}
        # for head_idx, pred in enumerate(preds):
        #     assert pred.shape[1] == 5
        #     bi, ki, zi = (0, 1, 2)
        #     pred = pred.permute(bi, zi, ki)
        #     pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

        #     label = forecast_labels[:, :, head_idx : head_idx + 1]
        #     auedit = metrics.distributed_AUED(pred, label)
        #     results = {
        #         f"val_{head_idx}_" + k: v for k, v in auedit.items()
        #     }
        #     step_result.update(results)
        step_result = {}
        for head_idx, pred in enumerate(preds):
            assert pred.shape[1] == k
            bi, ki, zi = (0, 1, 2)
            pred = pred.permute(bi, zi, ki)
            pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

            label = forecast_labels[:, :, head_idx : head_idx + 1]
            auedit = metrics.distributed_AUED(pred, label)
            results = {
                f"val_{head_idx}_" + k: v for k, v in auedit.items()
            }
            step_result.update(results)
        st()
        # torch.stack(preds_forecast).permute(0,3,2,1).squeeze(-1)
        # auedit = metrics.distributed_AUED(torch.stack(preds_forecast).permute(0,3,2,1).squeeze(-1), forecast_labels.permute(2,1,0))
        auedit = metrics.distributed_AUED(torch.sum(torch.stack(preds).permute(0,3,2,1).squeeze(-1), dim=0).unsqueeze(0), torch.sum(forecast_labels.permute(2,1,0), dim=0).unsqueeze(0))
        results = {
                f"sum_" + k: v for k, v in auedit.items()
            }
        step_result.update(results)
        step_result["response_idm"] = ""
        step_result["response_ltp"] = ""
        return step_result, []

    # def validation_step(self, batch, batch_idx):
    #     input, forecast_labels, _, _, label_clip_times = (
    #         batch
    #     )  # forecast_labels: (B, Z, 1)
    #     k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

    #     # Preds is a list of tensors of shape (B, K, Z), where
    #     # - B is batch size,
    #     # - K is number of predictions,
    #     # - Z is number of future predictions,
    #     # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])
    #     preds = self.model.generate(input, k=k)  # [(B, K, Z)]
    #     step_result = {}
    #     for head_idx, pred in enumerate(preds):
    #         assert pred.shape[1] == k
    #         bi, ki, zi = (0, 1, 2)
    #         pred = pred.permute(bi, zi, ki)
    #         pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

    #         label = forecast_labels[:, :, head_idx : head_idx + 1]
    #         auedit = metrics.distributed_AUED(pred, label)
    #         results = {
    #             f"val_{head_idx}_" + k: v for k, v in auedit.items()
    #         }
    #         step_result.update(results)

    #     return step_result

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        for key in keys:
            metric = torch.tensor([x[key] for x in outputs]).mean()
            self.log(key, metric)

    def test_step(self, batch, batch_idx):
        input, forecast_labels, video_labels, last_clip_ids, label_clip_times = (
            batch
        )  # forecast_labels: (B, Z, 1)
        k = self.cfg.FORECASTING.NUM_SEQUENCES_TO_PREDICT

        # Preds is a list of tensors of shape (B, K, Z), where
        # - B is batch size,
        # - K is number of predictions,
        # - Z is number of future predictions,
        # The list is for each label type (e.g. [<verb_tensor>, <noun_tensor>])

        # print()
        # print()
        # print("Video labels:")
        # for l_idx in range(video_labels.shape[1]):
        #     print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()
        # print("Forecasting labels:")
        # for l_idx in range(forecast_labels.shape[1]):
        #     print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
        # print()
        # print()
        # st()

        preds = self.model.generate(input, k=k)  # [(B, K, Z)]

        compute_metrics = True
        if not torch.all(forecast_labels==-1) and compute_metrics:
            step_result = {}
            for head_idx, pred in enumerate(preds):
                assert pred.shape[1] == k
                bi, ki, zi = (0, 1, 2)
                pred = pred.permute(bi, zi, ki)
                pred, forecast_labels = pred.cpu(), forecast_labels.cpu()

                label = forecast_labels[:, :, head_idx : head_idx + 1]
                auedit = metrics.distributed_AUED(pred, label)
                results = {
                    f"val_{head_idx}_" + k: v for k, v in auedit.items()
                }
                step_result.update(results)
            auedit = metrics.distributed_AUED(torch.sum(torch.stack(preds).permute(0,3,2,1).squeeze(-1), dim=0).unsqueeze(0).cpu(), torch.sum(forecast_labels.permute(2,1,0), dim=0).unsqueeze(0).cpu())
            results = {
                    f"sum_" + k: v for k, v in auedit.items()
                }
            step_result.update(results)
            # step_result["response_idm"] = ""
            # step_result["response_ltp"] = ""
            for k in step_result.keys():
                step_result[k] = float(step_result[k])
            self.results_dict[last_clip_ids[0]] = step_result
            with open(os.path.join('/home/gsarch/repo/forecasting/log/eval_validunseen_newsplit_1clip_slowfast_8x8_R101_06', 'metrics.json'), "w") as outfile: 
                json.dump(self.results_dict, outfile, indent=4, sort_keys=True)

        visualize = False
        if visualize:
            vid_num = 0
            std = self.cfg.DATA.STD
            mean = self.cfg.DATA.MEAN
            self.image_mean = torch.from_numpy(np.array([mean]).reshape(1,3,1,1)).cuda()
            self.image_std = torch.from_numpy(np.array([std]).reshape(1,3,1,1)).cuda()
            video_image = torch.cat(input[0][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
            video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
            video_image.save(f'images/test{vid_num}.png')
            video_image = torch.cat(input[1][0,vid_num].unbind(1), dim=2).unsqueeze(0) * self.image_std + self.image_mean
            video_image = Image.fromarray((video_image.squeeze(0).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8))
            video_image.save(f'images/test{vid_num}_full.png')
            print()
            print()
            print("Video labels:")
            for l_idx in range(video_labels.shape[1]):
                print(self.idx2verbs[int(video_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(video_labels[0,l_idx,1].cpu().numpy())])
            print()
            print()
            print("Forecasting labels:")
            for l_idx in range(forecast_labels.shape[1]):
                print(self.idx2verbs[int(forecast_labels[0,l_idx,0].cpu().numpy())], self.idx2nouns[int(forecast_labels[0,l_idx,1].cpu().numpy())])
            print()
            print()
            st()
        
        return {
            'last_clip_ids': last_clip_ids,
            'verb_preds': preds[0],
            'noun_preds': preds[1],
        }

    def test_epoch_end(self, outputs):

        test_outputs = {}
        for key in ['verb_preds', 'noun_preds']:
            preds = torch.cat([x[key] for x in outputs], 0)
            preds = self.all_gather(preds).unbind()
            test_outputs[key] = torch.cat(preds, 0)

        last_clip_ids = [x['last_clip_ids'] for x in outputs]
        last_clip_ids = [item for sublist in last_clip_ids for item in sublist]
        last_clip_ids = list(itertools.chain(*du.all_gather_unaligned(last_clip_ids)))
        test_outputs['last_clip_ids'] = last_clip_ids

        if du.get_local_rank() == 0:
            pred_dict = {}
            for idx in range(len(test_outputs['last_clip_ids'])):
                pred_dict[test_outputs['last_clip_ids'][idx]] = {
                    'verb': test_outputs['verb_preds'][idx].cpu().tolist(),
                    'noun': test_outputs['noun_preds'][idx].cpu().tolist(),
                }       
            json.dump(pred_dict, open('outputs.json', 'w'))


