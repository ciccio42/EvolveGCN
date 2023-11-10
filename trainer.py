import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
from iot23 import IoT23_Dataset
import os
from node_anomaly_tasker import Anomaly_Detection_Tasker
import models as mls
from Reconstruction_Loss import ReconstructionLoss
import wandb
from tqdm import tqdm


class Trainer():
    def __init__(self, args, splitter, gcn, classifier, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.gcn = gcn
        self.classifier = classifier
        self.comp_loss = comp_loss
        self.chpt_dir = os.path.join(args.save_folder, args.project_name)

        os.makedirs(self.chpt_dir, exist_ok=True)

        if not isinstance(dataset, IoT23_Dataset):
            self.num_nodes = dataset.num_nodes
        else:
            self.num_nodes = -1
        self.data = dataset
        self.num_classes = num_classes

        # self.logger = logger.Logger(args, self.num_classes)
        if self.args.wandb_log:
            # init wandb_log
            run = wandb.init(project=self.args.project_name,
                             name=self.args.project_name,
                             sync_tensorboard=False)

        self.init_optimizers(args)

        if self.tasker.is_static:
            adj_matrix = u.sparse_prepare_tensor(self.tasker.adj_matrix, torch_size=[
                                                 self.num_nodes], ignore_batch_dim=False)
            self.hist_adj_list = [adj_matrix]
            self.hist_ndFeats_list = [self.tasker.nodes_feats.float()]

    def init_optimizers(self, args):
        params = self.gcn.parameters()
        self.gcn_opt = torch.optim.Adam(params, lr=args.learning_rate)
        params = self.classifier.parameters()
        self.classifier_opt = torch.optim.Adam(params, lr=args.learning_rate)
        self.gcn_opt.zero_grad()
        self.classifier_opt.zero_grad()

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    def load_checkpoint(self, filename, model):
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            epoch = checkpoint['epoch']
            self.gcn.load_state_dict(checkpoint['gcn_dict'])
            self.classifier.load_state_dict(checkpoint['classifier_dict'])
            self.gcn_opt.load_state_dict(checkpoint['gcn_optimizer'])
            self.classifier_opt.load_state_dict(
                checkpoint['classifier_optimizer'])
            self.logger.log_str("=> loaded checkpoint '{}' (epoch {})".format(
                filename, checkpoint['epoch']))
            return epoch
        else:
            self.logger.log_str(
                "=> no checkpoint found at '{}'".format(filename))
            return 0

    def train(self):
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        for e in range(self.args.num_epochs):
            eval_train, nodes_embs = self.run_epoch(
                self.splitter.train, e, 'TRAIN', grad=True)
            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                eval_valid, _ = self.run_epoch(
                    self.splitter.dev, e, 'VALID', grad=False)

                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    print('### w'+str(self.args.rank)+') ep '+str(e) +
                          ' - Best valid measure:'+str(eval_valid))
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print('### w'+str(self.args.rank)+') ep ' +
                              str(e)+' - Early stop.')
                        break

            if len(self.splitter.test) > 0 and eval_valid == best_eval_valid and e > self.args.eval_after_epochs:
                eval_test, _ = self.run_epoch(
                    self.splitter.test, e, 'TEST', grad=False)

                if self.args.save_node_embeddings:
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.train_idx, log_file+'_train_nodeembs.csv.gz')
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.dev_idx, log_file+'_valid_nodeembs.csv.gz')
                    self.save_node_embs_csv(
                        nodes_embs, self.splitter.test_idx, log_file+'_test_nodeembs.csv.gz')

    def train_anomaly(self):
        self.tr_step = 0
        best_eval_valid = 0
        eval_valid = 0
        epochs_without_impr = 0

        tolog = {}
        for e in tqdm(range(self.args.num_epochs)):
            loss_train = self.run_epoch(
                self.splitter.train, e, 'TRAIN', grad=True)

            tolog['train_epoch'] = e
            tolog[f'train_epoch/anomaly_score'] = loss_train
            if self.args.wandb_log:
                wandb.log(tolog)

            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                eval_loss = self.run_epoch(
                    self.splitter.dev, e, 'VALID', grad=False)

                tolog['eval_epch'] = e
                tolog[f'eval_epoch/anomaly_score'] = eval_loss
                if self.args.wandb_log:
                    wandb.log(tolog)

                if eval_valid > best_eval_valid:
                    best_eval_valid = eval_valid
                    epochs_without_impr = 0
                    print('### w'+str(self.args.rank)+') ep '+str(e) +
                          ' - Best valid measure:'+str(eval_valid))
                    print(f'Saving best checkpoint epoch {e}')
                    torch.save(self.gcn.state_dict(), os.path.join(
                        self.chpt_dir, f"gnc_{e}.pt"))
                    torch.save(self.classifier.state_dict(), os.path.join(
                        self.chpt_dir, f"classifier_{e}.pt"))
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print('### w'+str(self.args.rank)+') ep ' +
                              str(e)+' - Early stop.')
                        break

    def run_epoch(self, split, epoch, set_name, grad):

        torch.set_grad_enabled(grad)
        tolog = {}
        loss_epoch = 0.0
        for indx, s in enumerate(split):
            step = (epoch+1) * (indx+1)
            # each split contains a
            if self.tasker.is_static:
                s = self.prepare_static_sample(s)
            else:
                if not isinstance(self.tasker, Anomaly_Detection_Tasker):
                    s = self.prepare_sample(s)
                else:
                    s = self.prepare_sample_anomaly(s)

            pred_res = self.predict(s.hist_adj_list,
                                    s.hist_ndFeats_list,
                                    s.label_sp,
                                    s.node_mask_list)
            if not isinstance(self.tasker, Anomaly_Detection_Tasker):
                predictions, nodes_embs = pred_res
                loss = self.comp_loss(predictions, s.label_sp['vals'])
            else:
                pred_attribute_list, pred_adj_list = pred_res
                if isinstance(self.comp_loss, ReconstructionLoss):
                    # pred_adj, gt_adj, pred_attri, gt_attri
                    loss = 0.0
                    for t in range(len(pred_adj_list)):
                        # compute the anomaly score for each timestamp
                        loss += self.comp_loss(pred_adj=pred_adj_list[t],
                                               gt_adj=s.hist_adj_list[t],
                                               pred_attri=pred_attribute_list[t],
                                               gt_attri=s.hist_ndFeats_list[t])

                    loss = torch.mean(loss/len(pred_adj_list))
                    loss_epoch += loss

            if set_name == 'VALID':
                tolog['val_step'] = step
                tolog[f'val_step/anomaly_score'] = loss
                # print(
                #     f"Validation epoch {epoch} - step {step}: Anomaly Score {loss}")
                if self.args.wandb_log:
                    wandb.log(tolog)

            elif set_name == "TRAIN":
                tolog['train_step'] = step
                tolog[f'train_step/anomaly_score'] = loss
                # print(
                #     f"Training epoch {epoch} - step {step}: Anomaly Score {loss}")
                if self.args.wandb_log:
                    wandb.log(tolog)

            if grad:
                self.optim_step(loss)

        torch.set_grad_enabled(True)

        return torch.mean(loss_epoch)

    def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, mask_list):
        nodes_embs = self.gcn(hist_adj_list,
                              hist_ndFeats_list,
                              mask_list)

        if not isinstance(self.classifier, mls.Decoder):
            predict_batch_size = 100000
            gather_predictions = []
            for i in range(1 + (node_indices.size(1)//predict_batch_size)):
                cls_input = self.gather_node_embs(
                    nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
                predictions = self.classifier(cls_input)
                gather_predictions.append(predictions)
            gather_predictions = torch.cat(gather_predictions, dim=0)
            return gather_predictions, nodes_embs
        else:
            # run decoder inference
            pred_attribute_list = []
            pred_adj_list = []
            for t, nodes_e in enumerate(nodes_embs):
                # adj_mat, feature_attribute
                pred_attribute_mat, pred_adj_mat = self.classifier(adj_mat=hist_adj_list[t],
                                                                   feature_attribute=nodes_e)
                pred_attribute_list.append(pred_attribute_mat)
                pred_adj_list.append(pred_adj_mat)

            return pred_attribute_list, pred_adj_list

    def gather_node_embs(self, nodes_embs, node_indices):
        cls_input = []

        for node_set in node_indices:
            cls_input.append(nodes_embs[node_set])
        return torch.cat(cls_input, dim=1)

    def optim_step(self, loss):
        self.tr_step += 1
        loss.backward()

        if self.tr_step % self.args.steps_accum_gradients == 0:
            self.gcn_opt.step()
            self.classifier_opt.step()

            self.gcn_opt.zero_grad()
            self.classifier_opt.zero_grad()

    def prepare_sample(self, sample):
        sample = u.Namespace(sample)
        for i, adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            # transposed to have same dimensions as scorer
            sample.node_mask_list[i] = node_mask.to(self.args.device).t()

        label_sp = self.ignore_batch_dim(sample.label_sp)

        if self.args.task in ["link_pred", "edge_cls"]:
            # ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
            label_sp['idx'] = label_sp['idx'].to(self.args.device).t()
        else:
            label_sp['idx'] = label_sp['idx'].to(self.args.device)

        label_sp['vals'] = label_sp['vals'].type(
            torch.long).to(self.args.device)
        sample.label_sp = label_sp

        return sample

    def prepare_sample_anomaly(self, sample):
        sample = u.Namespace(sample)
        for i, adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj, torch_size=[sample.n_nodes])
            sample.hist_adj_list[i] = adj.to(self.args.device)

            nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
            node_mask = sample.node_mask_list[i]
            # transposed to have same dimensions as scorer
            sample.node_mask_list[i] = node_mask.to(self.args.device).t()

            label_sp = self.ignore_batch_dim(sample.label_sp[i])

            if self.args.task in ["link_pred", "edge_cls"]:
                # ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
                label_sp['idx'] = label_sp['idx'].to(self.args.device).t()
            else:
                label_sp['idx'] = label_sp['idx'].to(self.args.device)

            label_sp['vals'] = label_sp['vals'].type(
                torch.long).to(self.args.device)
            sample.label_sp[i] = label_sp

        return sample

    def prepare_static_sample(self, sample):
        sample = u.Namespace(sample)

        sample.hist_adj_list = self.hist_adj_list

        sample.hist_ndFeats_list = self.hist_ndFeats_list

        label_sp = {}
        label_sp['idx'] = [sample.idx]
        label_sp['vals'] = sample.label
        sample.label_sp = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj

    def save_node_embs_csv(self, nodes_embs, indexes, file_name):
        csv_node_embs = []
        for node_id in indexes:
            orig_ID = torch.DoubleTensor(
                [self.tasker.data.contID_to_origID[node_id]])

            csv_node_embs.append(
                torch.cat((orig_ID, nodes_embs[node_id].double())).detach().numpy())

        pd.DataFrame(np.array(csv_node_embs)).to_csv(
            file_name, header=None, index=None, compression='gzip')
        # print ('Node embs saved in',file_name)
