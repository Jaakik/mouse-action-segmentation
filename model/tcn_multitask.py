#!/usr/bin/python2.7
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
from loguru import logger


# check

class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_1, num_classes_3):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes_1, num_classes_3)
        self.Rs_task1 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_1, num_classes_1)) for s in range(num_R)])

        self.Rs_beh_0 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_1 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_2 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_3 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_4 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_5 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])
        self.Rs_beh_6 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_3, num_classes_3)) for s in range(num_R)])

        self.Rs_ann_2 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_1, num_classes_1)) for s in range(num_R)])
        self.Rs_ann_3 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_1, num_classes_1)) for s in range(num_R)])
        self.Rs_ann_4 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_1, num_classes_1)) for s in range(num_R)])
        self.Rs_ann_5 = nn.ModuleList(
            [copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes_1, num_classes_1)) for s in range(num_R)])

        # task 2 will be a fully connected layer to predict the annotator for the whole seq

    def forward(self, x_1, x_2, x_3, behaviour, annotator):
        multitask = False
        out_1, out_2, out_3 = self.PG(x_1, x_2, x_3)
        outputs_1 = out_1.unsqueeze(0)
        outputs_2 = out_2.unsqueeze(0)
        outputs_3 = out_3.unsqueeze(0)

        # behaviour is a string
        # depending on which behaviour we would use a different head for prediction
        # adding an if statement to chose wheather to train for task 1 or 3
        for R in self.Rs_task1:
            out_1 = R(F.softmax(out_1, dim=1))
            outputs_1 = torch.cat((outputs_1, out_1.unsqueeze(0)), dim=0)

        ###
        if behaviour == "beh_0":
            for R in self.Rs_beh_1:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)
        if behaviour == "beh_1":
            for R in self.Rs_beh_1:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        if behaviour == "beh_2":
            for R in self.Rs_beh_2:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        if behaviour == "beh_3":
            for R in self.Rs_beh_3:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        if behaviour == "beh_4":
            for R in self.Rs_beh_3:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        if behaviour == "beh_5":
            for R in self.Rs_beh_1:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        if behaviour == "beh_6":
            for R in self.Rs_beh_1:
                out_3 = R(F.softmax(out_3, dim=1))
                outputs_3 = torch.cat((outputs_3, out_3.unsqueeze(0)), dim=0)

        ########################        ########################        ########################

        if annotator == "ann_1":
            for R in self.Rs_task1:
                out_2 = R(F.softmax(out_2, dim=1))
                outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(0)), dim=0)

        if annotator == "ann_2":
            for R in self.Rs_ann_2:
                out_2 = R(F.softmax(out_2, dim=1))
                outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(0)), dim=0)

        if annotator == "ann_3":
            for R in self.Rs_ann_3:
                out_2 = R(F.softmax(out_2, dim=1))
                outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(0)), dim=0)

        if annotator == "ann_4":
            for R in self.Rs_ann_4:
                out_2 = R(F.softmax(out_2, dim=1))
                outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(0)), dim=0)

        if annotator == "ann_5":
            for R in self.Rs_ann_5:
                out_2 = R(F.softmax(out_2, dim=1))
                outputs_2 = torch.cat((outputs_2, out_2.unsqueeze(0)), dim=0)

        return outputs_1, outputs_2, outputs_3


class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes_1, num_classes_3):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** (num_layers - 1 - i), dilation=2 ** (num_layers - 1 - i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2 ** i, dilation=2 ** i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
            nn.Conv1d(2 * num_f_maps, num_f_maps, 1)
            for i in range(num_layers)

        ))

        self.dropout = nn.Dropout()

        # if task 3 or task 1 separate cases and return 2 vectors
        self.conv_out_1 = nn.Conv1d(num_f_maps, num_classes_1, 1)
        # initial binary predictions that will go into refinement
        self.conv_out_3 = nn.Conv1d(num_f_maps, num_classes_3, 1)

    def forward(self, x_1, x_2, x_3):
        f_1 = self.conv_1x1_in(x_1)
        for i in range(self.num_layers):
            f_in = f_1
            f_1 = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f_1), self.conv_dilated_2[i](f_1)], 1))
            f_1 = F.relu(f_1)
            f_1 = self.dropout(f_1)
            f_1 = f_1 + f_in

        f_2 = self.conv_1x1_in(x_2)
        for i in range(self.num_layers):
            f_in = f_2
            f_2 = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f_2), self.conv_dilated_2[i](f_2)], 1))
            f_2 = F.relu(f_2)
            f_2 = self.dropout(f_2)
            f_2 = f_2 + f_in

        f_3 = self.conv_1x1_in(x_3)
        for i in range(self.num_layers):
            f_in = f_3
            f_3 = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f_3), self.conv_dilated_2[i](f_3)], 1))
            f_3 = F.relu(f_3)
            f_3 = self.dropout(f_3)
            f_3 = f_3 + f_in

        out_1 = self.conv_out_1(f_1)
        out_2 = self.conv_out_1(f_2)
        out_3 = self.conv_out_3(f_3)

        return out_1, out_2, out_3


class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList(
            [copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages - 1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_1, num_classes_3):
        self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes_1=num_classes_1,
                             num_classes_3=num_classes_3)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes_1 = num_classes_1
        self.num_classes_3 = num_classes_3

        logger.add('logs/' + "_" + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct_1 = 0
            total_1 = 0
            correct_2 = 0
            total_2 = 0
            correct_3 = 0
            total_3 = 0
            while batch_gen.has_next():
                batch_input_1, batch_target_1, mask_1, batch_input_2, batch_target_2, mask_2, batch_input_3, batch_target_3, mask_3, behaviour, annotator = batch_gen.next_batch(
                    batch_size)
                batch_input_1, batch_target_1, mask_1 = batch_input_1.to(device), batch_target_1.to(device), mask_1.to(
                    device)
                batch_input_2, batch_target_2, mask_2 = batch_input_2.to(device), batch_target_2.to(device), mask_2.to(
                    device)
                batch_input_3, batch_target_3, mask_3 = batch_input_3.to(device), batch_target_3.to(device), mask_3.to(
                    device)

                optimizer.zero_grad()
                # multitask learning
                predictions_1, predictions_2, predictions_3 = self.model(batch_input_1, batch_input_2, batch_input_3,
                                                                         behaviour, annotator)

                loss_1 = 0
                loss_2 = 0
                loss_3 = 0

                # TODO weight losses for finetuning
                w1 = 0.4
                w2 = 0.4
                w3 = 0.3

                for p in predictions_1:
                    loss_1 += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_1),
                                      batch_target_1.view(-1))
                    loss_1 += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask_1[:, :, 1:])

                for p in predictions_2:
                    loss_2 += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_1),
                                      batch_target_2.view(-1))
                    loss_2 += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask_2[:, :, 1:])

                for p in predictions_3:
                    loss_3 += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes_3),
                                      batch_target_3.view(-1))
                    loss_3 += 0.15 * torch.mean(torch.clamp(
                        self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                        max=16) * mask_3[:, :, 1:])

                loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted_1 = torch.max(predictions_1[-1].data, 1)
                _, predicted_2 = torch.max(predictions_2[-1].data, 1)
                _, predicted_3 = torch.max(predictions_3[-1].data, 1)

                correct_1 += ((predicted_1 == batch_target_1).float() * mask_1[:, 0, :].squeeze(1)).sum().item()
                total_1 += torch.sum(mask_1[:, 0, :]).item()

                correct_2 += ((predicted_2 == batch_target_2).float() * mask_2[:, 0, :].squeeze(1)).sum().item()
                total_2 += torch.sum(mask_2[:, 0, :]).item()

                correct_3 += ((predicted_3 == batch_target_3).float() * mask_3[:, 0, :].squeeze(1)).sum().item()
                total_3 += torch.sum(mask_3[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("loss_1")
            print(loss_1.item())
            print("loss_2")
            print(loss_2.item())
            print("loss_3")
            print(loss_3.item())
            logger.info(
                "[epoch %d]: epoch loss = %f,   acc_1 = %f,acc_3 = %f, acc_3 = %f " % (
                    epoch + 1, epoch_loss / 3 * len(batch_gen.list_of_examples_1),
                    float(correct_1) / total_1, float(correct_2) / total_2, float(correct_3) / total_3))
