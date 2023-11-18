import torch
import utils as u
import logger
import time
import pandas as pd
import numpy as np
from anomaly import Anomaly_Dataset
import os
from node_anomaly_tasker import Anomaly_Detection_Tasker
import models as mls
from Reconstruction_Loss import ReconstructionLoss
import wandb
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import json
import gc


class TrainerAnomaly():
    def __init__(self, args, splitter, detector, comp_loss, dataset, num_classes):
        self.args = args
        self.splitter = splitter
        self.tasker = splitter.tasker
        self.detector = detector
        self.comp_loss = comp_loss
        self.chpt_dir = os.path.join(args.save_folder, args.project_name)

        os.makedirs(self.chpt_dir, exist_ok=True)

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

    def init_optimizers(self, args):
        self.detector_opt = torch.optim.Adam(
            self.detector.parameters(), lr=args.learning_rate)
        self.detector_opt.zero_grad()

    def train_anomaly(self):
        self.tr_step = 0
        best_eval_loss = np.inf
        epochs_without_impr = 0

        tolog = {}
        for e in tqdm(range(self.args.num_epochs)):
            loss_epoch, attr_error_epoch, stru_error_epoch = self.run_epoch(
                self.splitter.train, e, 'TRAIN', grad=True)

            tolog['train_epoch'] = e
            tolog[f'train_epoch/anomaly_score'] = loss_epoch
            tolog[f'train_epoch/attribute_error'] = attr_error_epoch
            tolog[f'train_epoch/structure_error'] = stru_error_epoch
            if self.args.wandb_log:
                wandb.log(tolog)

            if len(self.splitter.dev) > 0 and e > self.args.eval_after_epochs:
                loss_epoch, attr_error_epoch, stru_error_epoch = self.run_epoch(
                    self.splitter.dev, e, 'VALID', grad=False)

                tolog['eval_epoch'] = e
                tolog[f'val_epoch/anomaly_score'] = loss_epoch
                tolog[f'val_epoch/attribute_error'] = attr_error_epoch
                tolog[f'val_epoch/structure_error'] = stru_error_epoch
                if self.args.wandb_log:
                    wandb.log(tolog)

                if loss_epoch < best_eval_loss:
                    best_eval_loss = loss_epoch
                    epochs_without_impr = 0
                    print('### w'+str(self.args.rank)+') ep '+str(e) +
                          ' - Best valid measure:'+str(loss_epoch))
                    print(f'Saving best checkpoint epoch {e}')
                    torch.save(self.detector, os.path.join(
                        self.chpt_dir, f"detector_{e}.pt"))
                    torch.save(self.detector_opt.state_dict(), os.path.join(
                        self.chpt_dir, f"detector_opt_{e}.pt"))
                else:
                    epochs_without_impr += 1
                    if epochs_without_impr > self.args.early_stop_patience:
                        print('### w'+str(self.args.rank)+') ep ' +
                              str(e)+' - Early stop.')
                        break

    def test_anomaly(self, compute_thr=True):
        tolog = {}
        if compute_thr:
            print(f"Computing threshold...")
            # 1. Run inference on validation set
            # labels have indx and value
            eval_scores, labels = self.run_epoch(
                self.splitter.dev, -1, 'VALID', grad=False)
            # save scores
            os.makedirs(os.path.join(self.chpt_dir,
                        "threshold"), exist_ok=True)

            with open(os.path.join(self.chpt_dir, "threshold", "score_list.pkl"), "wb") as handle:
                pickle.dump(eval_scores, handle, protocol=4)

            with open(os.path.join(self.chpt_dir, "threshold", "label_list.pkl"), "wb") as handle:
                pickle.dump(labels, handle, protocol=4)

            # 2. Find the best threshold
            threshold = self.find_best_threshold()

        if not compute_thr:
            # load best threshold
            file_path = os.path.join(
                self.chpt_dir, "threshold", "optimal_threshold.txt")
            with open(file_path, 'r') as file:
                threshold = float(file.read())

        # self.run_test(split_name='validation',
        #               threshold=threshold)
        # self.run_test(split_name='iot_traces',
        #               threshold=threshold)
        if not self.tasker.data.sequence:

            self.run_test(split_name='test_benign',
                          threshold=threshold)
            self.run_test(split_name='test_malicious',
                          threshold=threshold)
            self.run_test(split_name='test_mixed',
                          threshold=threshold)

            self.run_test(split_name='iot_id20_benign',
                          threshold=threshold)
            self.run_test(split_name='iot_id20_mixed',
                          threshold=threshold)
        else:
            self.run_test(split_name='test_iot23',
                          threshold=threshold)
            self.run_test(split_name='test_iot_id20',
                          threshold=threshold)

    def run_test(self, split_name, threshold):
        tolog = {}
        # 3. Test with best threshold
        print(f"Running test on {split_name}....")
        if split_name == 'validation':
            epoch_name = None
            split = self.splitter.dev
        elif split_name == 'test_benign':
            epoch_name = None
            split = self.splitter.test_benign
        elif split_name == 'test_malicious':
            epoch_name = None
            split = self.splitter.test_malicious
        elif split_name == 'test_mixed':
            epoch_name = None
            split = self.splitter.test_mixed
        elif split_name == 'iot_traces':
            epoch_name = None
            split = self.splitter.test_traces
        elif split_name == 'iot_id20_benign':
            epoch_name = None
            split = self.splitter.test_iotid20_benign
        elif split_name == 'iot_id20_mixed':
            epoch_name = None
            split = self.splitter.test_iotid20_mixed
        elif split_name == 'test_iot23':
            epoch_name = None
            split = self.splitter.test_iot23
        elif split_name == 'test_iot_id20':
            epoch_name = None
            split = self.splitter.test_iotid20

        eval_scores, labels = self.run_epoch(
            split, -1, epoch_name, grad=False, test=True)

        # Compute validation metrics
        accuracy, precision, recall, f_score, tp, tn, fp, fn = self.compute_metrics(
            y_scores=eval_scores,
            y_labels=labels,
            threshold=threshold)
        tolog[f'{split_name}/accuracy'] = accuracy
        tolog[f'{split_name}/precision'] = precision
        tolog[f'{split_name}/recall'] = recall
        tolog[f'{split_name}/f_score'] = f_score
        tolog[f'{split_name}/tp'] = tp
        tolog[f'{split_name}/tn'] = tn
        tolog[f'{split_name}/fp'] = fp
        tolog[f'{split_name}/fn'] = fn
        with open(f'{self.chpt_dir}/{split_name}.json', 'w') as fp:
            json.dump(tolog, fp)
        if self.args.wandb_log:
            wandb.log(tolog)

    def compute_metrics(self, y_scores, y_labels, threshold):
        # create y_pred
        y_pred = []
        y_true = []

        # number of predictions
        n_pred = 0.0

        for b in range(len(y_scores)):
            for t in range(len(y_scores[b])):
                y_pred.append(y_scores[b][t][y_labels[b][t][0][0]] > threshold)
                y_true.append(y_labels[b][t][1])
                n_pred += y_labels[b][t][1].shape[0]

        tp = 0
        tn = 0
        fn = 0
        fp = 0
        try:
            for i in range(len(y_pred)):
                for n_indx in range(y_pred[i].shape[0]):
                    if y_pred[i][n_indx] == 1 and y_true[i][n_indx] == 1:
                        tp += 1
                    if y_pred[i][n_indx] == 0 and y_true[i][n_indx] == 0:
                        tn += 1
                    if y_pred[i][n_indx] == 1 and y_true[i][n_indx] == 0:
                        fp += 1
                    if y_pred[i][n_indx] == 0 and y_true[i][n_indx] == 1:
                        fn += 1

            # print("\nTP:", tp)
            # print("\nTN:", tn)
            # print("\nFP:", fp)
            # print("\nFN:", fn)

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * (precision * recall) / (precision + recall)
        except:
            precision = 0
            recall = 0
            f_score = 0

        return accuracy, precision, recall, f_score, tp, tn, fp, fn

    def find_best_threshold(self):
        # load scores and labels
        with open(os.path.join(self.chpt_dir, "threshold", "score_list.pkl"), "rb") as handle:
            eval_scores = pickle.load(handle)  # B x T x Nodes

        with open(os.path.join(self.chpt_dir, "threshold", "label_list.pkl"), "rb") as handle:
            labels = pickle.load(handle)

        # 1. Find the max score
        max_score = self.find_max_score(eval_scores, labels)
        threshold = max_score/2
        optimal_threshold, threshold_list, fp_list = self.find_optimal_threshold(
            y_scores=eval_scores,
            y_labels=labels,
            threshold=threshold)
        self.create_fp_vs_threshold_plot(save_path=self.chpt_dir,
                                         fp_list=fp_list,
                                         threshold_list=threshold_list)

        # save optimal_threshold
        print("Optimal threshold to use in the next step: ", optimal_threshold)
        threshold_file_path = os.path.join(
            self.chpt_dir, "threshold", "optimal_threshold.txt")
        with open(threshold_file_path, "w") as f:
            f.write(str(optimal_threshold.item()))

        return optimal_threshold.item()

    def find_max_score(self, y_scores, y_labels):
        max_score = -np.inf

        for b in range(len(y_scores)):
            for t in range(len(y_scores[b])):
                max_score_for_sample = torch.max(
                    y_scores[b][t][y_labels[b][t][0][0]])
                if max_score_for_sample > max_score:
                    max_score = max_score_for_sample

        return max_score

    def create_fp_vs_threshold_plot(self, save_path, fp_list, threshold_list):
        plt.figure(figsize=(10, 6))
        plt.plot(threshold_list, fp_list, marker='o',
                 linestyle='-', color='b')

        plt.title('False Positive vs Threshold')
        plt.xlabel('Threshold')
        plt.ylabel('False Positive')

        plt.grid(True)
        plt.savefig(os.path.join(
            save_path, 'fp_vs_threshold.png'), format='png', dpi=300)

    def find_optimal_threshold(self, y_scores, y_labels, threshold):

        def find_fp(y_true, y_pred):
            # Positive is anomaly node
            # Fp is a normale node (label 0) that is label as anomaly (label 1)
            fp = 0
            for s in range(len(y_pred)):
                true_negative_indx = y_true[s] == 0
                # sample elements true_negative nodes, predicted as 1
                fp_pred = y_pred[s][true_negative_indx] == 1
                fp += torch.count_nonzero(fp_pred)

            return fp

        # create y_pred
        y_pred = []
        y_true = []
        threshold_list = []
        fp_list = []
        # number of predictions
        n_pred = 0.0
        for b in range(len(y_scores)):
            for t in range(len(y_scores[b])):
                y_pred.append(y_scores[b][t][y_labels[b][t][0][0]] > threshold)
                y_true.append(y_labels[b][t][1])
                n_pred += y_labels[b][t][1].shape[0]

        fp = find_fp(y_true, y_pred)
        fp_percentage = (fp / n_pred) * 100
        threshold_list.append(threshold)
        fp_list.append(fp)
        final_threshold = threshold
        while fp_percentage > 1:
            if not self.data.normalize:
                final_threshold += 1000
            else:
                final_threshold += 0.5
            print(final_threshold)
            y_pred = []
            for b in range(len(y_scores)):
                for t in range(len(y_scores[b])):
                    y_pred.append(
                        y_scores[b][t][y_labels[b][t][0][0]] > threshold)
            fp = find_fp(y_true, y_pred)
            print(fp)
            fp_percentage = (fp / n_pred) * 100
            print(fp_percentage)
            threshold_list.append(threshold)
            fp_list.append(fp)
            if 0.9 < fp_percentage < 1:
                break
        print(f"False-positive {fp_percentage}")
        return final_threshold, threshold_list, fp_list

    def run_epoch(self, split, epoch, set_name, grad, test=False):

        torch.set_grad_enabled(grad)
        tolog = {}
        loss_epoch = 0.0
        attr_error_epoch = 0.0
        stru_error_epoch = 0.0
        scores_epoch = []
        labels_epoch = []
        for indx, s_list in enumerate(tqdm(split)):
            B = len(s_list)

            step = (epoch+1) * (indx+1)
            for b_indx in range(B):
                # each split contains a
                s = self.prepare_sample_anomaly(s_list[b_indx])

                pred_res = self.predict(s.hist_adj_list_norm,
                                        s.hist_ndFeats_list,
                                        s.label_sp,
                                        s.node_mask_list)

                pred_attribute_list, pred_adj_list = pred_res
                if isinstance(self.comp_loss, ReconstructionLoss) and epoch != -1:
                    # pred_adj, gt_adj, pred_attri, gt_attri
                    loss_node = 0.0
                    attr_error_node = 0.0
                    stru_error_node = 0.0
                    loss = 0.0
                    attr_error = 0.0
                    stru_error = 0.0
                    for t in range(len(pred_adj_list)):
                        # compute the anomaly score for each timestamp
                        loss_t, attr_error_t, stru_error_t = self.comp_loss(pred_adj=pred_adj_list[t],
                                                                            gt_adj=s.hist_adj_list[t],
                                                                            pred_attri=pred_attribute_list[t],
                                                                            gt_attri=s.hist_ndFeats_list[t],
                                                                            node_mask=s.node_mask_list[t],
                                                                            partial_mat=s.hist_adj_list_partial[t],
                                                                            test=test)
                        loss_node += loss_t
                        attr_error_node += attr_error_t
                        stru_error_node += stru_error_t

                    loss += torch.mean(loss_node/len(pred_adj_list))
                    attr_error += torch.mean(attr_error_node /
                                             len(pred_adj_list))
                    stru_error += torch.mean(stru_error_node /
                                             len(pred_adj_list))

                    loss_epoch += loss
                    attr_error_epoch += attr_error
                    stru_error_epoch += stru_error
                else:
                    scores_step = []
                    labels_step = []
                    for t in range(len(pred_adj_list)):
                        with torch.no_grad():
                            loss_t, _, _ = self.comp_loss(pred_adj=pred_adj_list[t],
                                                          gt_adj=s.hist_adj_list[t],
                                                          pred_attri=pred_attribute_list[t],
                                                          gt_attri=s.hist_ndFeats_list[t],
                                                          node_mask=s.node_mask_list[t],
                                                          partial_mat=s.hist_adj_list_partial[t],
                                                          test=test)
                            scores_step.append(loss_t.cpu())
                            labels_step.append(
                                [s.label_sp[t]['idx'].cpu(), s.label_sp[t]['vals'].cpu()])

                    scores_epoch.append(scores_step)
                    labels_epoch.append(labels_step)

                    del pred_attribute_list, pred_adj_list
                    gc.collect()
                    torch.cuda.empty_cache()

            if set_name == 'VALID' and epoch != -1:
                tolog['val_step'] = step
                tolog[f'val_step/anomaly_score'] = loss
                tolog[f'val_step/attr_error'] = attr_error
                tolog[f'val_step/stru_error'] = stru_error
                # print(
                #     f"Validation epoch {epoch} - step {step}: Anomaly Score {loss}")
                if self.args.wandb_log:
                    wandb.log(tolog)

            elif set_name == "TRAIN":
                tolog['train_step'] = step
                tolog[f'train_step/anomaly_score'] = loss
                tolog[f'train_step/attr_error'] = attr_error
                tolog[f'train_step/stru_error'] = stru_error
                # print(
                #     f"Training epoch {epoch} - step {step}: Anomaly Score {loss}")
                if self.args.wandb_log:
                    wandb.log(tolog)

            if grad:
                self.optim_step(loss)

        torch.set_grad_enabled(True)
        if epoch != -1:
            return loss_epoch/indx, attr_error_epoch/indx, stru_error_epoch/indx
        else:
            return scores_epoch, labels_epoch

    def predict(self, hist_adj_list, hist_ndFeats_list, node_indices, mask_list):
        return self.detector(hist_adj_list=hist_adj_list,
                             hist_ndFeats_list=hist_ndFeats_list,
                             mask_list=mask_list)

    def optim_step(self, loss):
        tolog = dict()
        self.tr_step += 1
        # a = list(self.detector.parameters())[-1].clone()
        loss.backward()
        # if self.tr_step % self.args.steps_accum_gradients == 0:
        self.detector_opt.step()
        self.detector_opt.zero_grad()
        # b = list(self.detector.parameters())[-1].clone()
        # print(torch.equal(a.data, b.data))

    def prepare_sample_anomaly(self, sample):

        sample = u.Namespace(sample)

        for i, adj in enumerate(sample.hist_adj_list):
            adj = u.sparse_prepare_tensor(adj, torch_size=[sample.n_nodes],
                                          ignore_batch_dim=False)
            sample.hist_adj_list[i] = adj.to(self.args.device)

            adj_norm = u.sparse_prepare_tensor(
                sample.hist_adj_list_norm[i],
                torch_size=[sample.n_nodes],
                ignore_batch_dim=False)
            sample.hist_adj_list_norm[i] = adj_norm.to(
                self.args.device)

            adj_partial = torch.FloatTensor(sample.hist_adj_list_partial[i])
            sample.hist_adj_list_partial[i] = adj_partial.to(self.args.device)

            # nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])
            nodes = torch.FloatTensor(sample.hist_ndFeats_list[i])
            sample.hist_ndFeats_list[i] = nodes.to(self.args.device)

            node_mask = torch.FloatTensor(sample.node_mask_list[i])
            # transposed to have same dimensions as scorer
            sample.node_mask_list[i] = node_mask.to(self.args.device).t()

            # self.ignore_batch_dim(sample.label_sp[i])
            label_sp = sample.label_sp[i]

            label_sp['idx'] = label_sp['idx'].to(self.args.device)

            label_sp['vals'] = label_sp['vals'].type(
                torch.long).to(self.args.device)
            sample.label_sp[i] = label_sp

        return sample

    def ignore_batch_dim(self, adj):
        if self.args.task in ["link_pred", "edge_cls"]:
            adj['idx'] = adj['idx'][0]
        adj['vals'] = adj['vals'][0]
        return adj
