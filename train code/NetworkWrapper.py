import copy

import itertools

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from IPython.display import clear_output
from torch.utils.data import Dataset
from copy import deepcopy
import cv2
from torch.utils.tensorboard import SummaryWriter




class NN(nn.Module):
    def __init__(self, network):
        super(NN, self).__init__()
        self.model = network

    def forward(self, data):
        return self.model(data)


class DefinedNN:
    def __init__(self, neural_network, n_classes, loss_function, optimiser, lr_scheduler=None, device="cpu"):
        self.sorted_predictions = None
        self.neural_network = neural_network
        self.loss_func = loss_function
        self.optimiser = optimiser
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.train_acc = []
        self.lr_scheduler = lr_scheduler
        self.states = []
        self.best_states = None
        self.n_classes = n_classes

        self.device = torch.device(device)

        self.classes = DefinedNN.classes()
        self.classes_dictionary = DefinedNN.classes_dictionary()
        self.tensorboard = False
        self.tensorboard_writer = None

    def train(self, train_set, val_set, **kwargs):
        n_workers = kwargs.pop("workers", 0)
        batch_size = kwargs.pop("batch_size", 100)
        max_epoch = kwargs.pop("max_epoch", 50)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)
        valloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=n_workers)

        network = self.neural_network.to(self.device)
        if self.tensorboard:
            self.tensorboard_writer.add_graph(self.neural_network, next(iter(trainloader))[0].to(self.device))

        print("Iteration Progress: [%d/%d]" % (0, max_epoch))
        for epoch in range(max_epoch):
            running_loss = 0.0
            running_correct = 0
            for sample, label in trainloader:
                self.neural_network.train()
                sample, label = sample.to(self.device), label.to(self.device)
                x_hat = network(sample)
                loss = self.loss_func(x_hat, label.long())

                running_loss += loss.item()
                running_correct += (torch.max(x_hat, 1)[1] == label).sum().item()

                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
            self.train_loss.append(running_loss / train_set.labels.shape[0])
            self.train_acc.append(running_correct / train_set.labels.shape[0])

            with torch.no_grad():
                self.neural_network.eval()
                running_loss = 0.0
                running_correct = 0
                for sample, label in valloader:
                    sample, label = sample.to(self.device), label.to(self.device)

                    val_y_hat = network(sample)
                    loss = self.loss_func(val_y_hat, label.long())

                    running_loss += loss.item()
                    running_correct += (torch.max(val_y_hat, 1)[1] == label).sum().item()

                self.val_loss.append(running_loss / val_set.labels.shape[0])
                self.val_acc.append(running_correct / val_set.labels.shape[0])

            if self.tensorboard:
                self.tensorboard_writer.add_scalar('Loss/Train', self.train_loss[-1], epoch)
                self.tensorboard_writer.add_scalar('Loss/Validation', self.val_loss[-1], epoch)
                self.tensorboard_writer.add_scalar('Accuracy/Train', self.train_acc[-1], epoch)
                self.tensorboard_writer.add_scalar('Accuracy/Validation', self.val_acc[-1], epoch)
                self.tensorboard_writer.flush()


            if self.lr_scheduler:
                self.lr_scheduler.step()
            self.states.append(copy.deepcopy(self.neural_network.cpu().state_dict()))
            self.neural_network = self.neural_network.to(self.device)
            clear_output(True)
            print("Iteration Progress: [%d/%d]" % (epoch + 1, max_epoch))

        self.__extract_best()

        return

    def __extract_best(self):
        val_loss_argmin = np.argmin(self.val_loss)
        train_loss_argmin = np.argmin(self.train_loss)
        val_acc_argmax = np.argmax(self.val_acc)

        print(f"Maxmimum Validation Accuracy:{self.val_acc[val_acc_argmax]}")

        best_states = {"val_loss": self.states[val_loss_argmin], "train_loss": self.states[train_loss_argmin], "val_acc": self.states[val_acc_argmax]}

        if self.tensorboard:
            if self.lr_scheduler:
                lr_exists = True
                lr_step = int(self.lr_scheduler.step_size)
                lr_gamma = float(self.lr_scheduler.gamma)
            else:
                lr_exists = False
                lr_step = False
                lr_gamma = False

            try:
                momentum = self.optimiser.defaults["momentum"]
            except KeyError:
                momentum = 0
            try:
                dampening = self.optimiser.defaults["dampening"]
            except KeyError:
                dampening = 0

            self.tensorboard_writer.add_hparams({'Optimiser': type(self.optimiser).__name__,
                                                 'Initial LR': float(self.optimiser.defaults["lr"]),
                                                 'Weight Decay': float(self.optimiser.defaults["weight_decay"]),
                                                 'Momentum': momentum,
                                                 'Dampening': dampening,
                                                 'LR Scheduling': lr_exists,
                                                 'LR Scheduler Gamma': lr_gamma,
                                                 'LR Scheduler Step Size': lr_step},
                                                {'hparam/Validation Accuracy': float(self.val_acc[val_acc_argmax]),
                                                 'hparam/Validation Loss': float(self.val_loss[val_acc_argmax])})

            self.tensorboard_writer.flush()

        self.best_states = best_states

    def __get_lr(self):
        if self.lr_scheduler:
            return self.lr_scheduler.get_lr()
        else:
            for parameter_group in self.optimiser.param_groups:
                return parameter_group['lr']

    def start_tensorboard(self, runs_dir):
        self.tensorboard = True
        self.tensorboard_writer = SummaryWriter(runs_dir)

    def add_image(self, img, title):
        self.tensorboard_writer.add_image(title, img)

    def add_embedding(self, **kwargs):
        self.tensorboard_writer.add_embedding(**kwargs)



    def graph(self):
        plot_fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.val_acc, label="Validation Accuracy", color="green")
        ax2.plot(self.val_loss, label="Validation Loss", color="green")

        ax1.plot(self.train_acc, label="Training Accuracy", color="blue")
        ax2.plot(self.train_loss, label="Training Loss", color="blue")

        plt.suptitle(f"Accuracy and Loss vs epoch\nlr: {self.__get_lr()}\nOptimiser: {type(self.optimiser).__name__}")

        ax1.set_xlabel("epoch")
        ax2.set_xlabel("epoch")

        ax1.set_ylabel("Accuracy")
        ax2.set_ylabel("Loss")

        ax1.set_title("Accuracy")
        ax2.set_title("Loss")

        handles, _ = ax1.get_legend_handles_labels()
        plot_fig1.legend(handles, ["Validation", "Training"], loc='upper right')

        return plot_fig1

    def test_model(self, dataset):
        sorted_predictions = [{True: [], False: []},
                              {True: [], False: []},
                              {True: [], False: []},
                              {True: [], False: []},
                              {True: [], False: []},
                              {True: [], False: []},
                              {True: [], False: []}]

        # Index of sorted_predictions represent the label.
        # True array contains a dictionary with keys of softmax_score and its sample tensor and label where the prediction has correctly predicted.
        # Likewise for False array.

        with torch.no_grad():
            self.neural_network.eval()
            loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=0)
            running_correct = 0
            for samples, labels in loader:
                samples, labels = samples.to(self.device), labels.to(self.device)
                y_hat = self.neural_network(samples)

                scores = F.softmax(y_hat, dim=1)
                (maximum, max_indices) = torch.max(scores, dim=1)
                correct_tensor = max_indices == labels
                running_correct += correct_tensor.sum().item()

                for softmax_score, index, correct, sample, true_label in zip(scores, max_indices, correct_tensor, samples, labels):
                    target = sorted_predictions[index].get(correct.item())

                    target.append({"label": true_label.cpu().item(), "score": softmax_score[index].item(), "sample": sample.cpu()})
            print(running_correct / dataset.labels.shape[0])
            self.sorted_predictions = sorted_predictions
        return sorted_predictions

    def test_single(self, data, label):
        with torch.no_grad():
            self.neural_network.eval()
            score = F.softmax(self.neural_network(torch.Tensor(np.expand_dims(data.numpy(), axis=0)).to(self.device)), dim=1)

        return {"label": label, "score": score[0][label].item(), "sample": data, "classified_label": torch.argmax(score, dim=1)}

    def __calc_confusion_matrix(self):
        if not self.sorted_predictions:
            raise AssertionError("Confusion Matrix can only be calculated once the model as been tested.")

        confusion_matrix = np.zeros((self.n_classes, self.n_classes))

        for index, arr in enumerate(self.sorted_predictions):
            for key in arr.keys():

                target = confusion_matrix[index, :]
                for sample in arr.get(key):
                    target[sample.get("label")] += 1

        return confusion_matrix

    def visualise_confusion_matrix(self, cmap):
        """
        Source: Taken from DEEPLIZARD.com and modified.
        URL: https://deeplizard.com/learn/video/0LhiS6yu2qQ

        Graphs the confusion matrix
        :param cmap: Colour mapping
        :param classes: Array of class names
        :return: NA

        """
        fig = plt.figure(figsize=(10, 10))
        confusion_matrix = self.__calc_confusion_matrix()

        plt.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        fmt = '.2f'
        thresh = confusion_matrix.max() / 2.
        for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
            plt.text(j, i, format(confusion_matrix[i, j], fmt), horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return fig

    def load_model(self, model='val_acc'):
        self.neural_network = self.neural_network.cpu()
        self.neural_network.load_state_dict(self.best_states[model])
        self.neural_network = self.neural_network.to(self.device)

    def occlusion_sensitivity(self, image_dic, kernel=(5, 5), factor=255):
        image = image_dic.get("sample")
        label = image_dic.get("label")
        original_score = image_dic.get("score")
        image_score = image
        if len(image.squeeze().shape) == 2:
            image_score = torch.cat((image, image, image), dim=0)

        if image.shape[1] % kernel[0] != 0 or image.shape[2] % kernel[1] != 0:
            raise ValueError("Kernel must be a factor of image width and height.")

        score_matrix = []
        image_score_heatmap = deepcopy(image_score.numpy())
        image_history = [image]

        image_height = image.shape[1]
        image_width = image.shape[2]

        width_diff = image_width - kernel[0]
        height_diff = image_height - kernel[1]

        with torch.no_grad():
            self.neural_network.eval()
            for height_loc in range(0, height_diff + 1, kernel[1]):
                score_matrix.append([])
                score_row = score_matrix[-1]
                for width_loc in range(0, width_diff + 1, kernel[0]):
                    image_copy = deepcopy(image)
                    image_copy[:, height_loc:height_loc + kernel[1], width_loc:width_loc + kernel[0]] = 0
                    image_history.append(image_copy)

                    score = F.softmax(self.neural_network(image.unsqueeze(1).to(self.device)), dim=1)
                    score_row.append(original_score - score[0][label].item())
                    image_score_heatmap[0, height_loc:height_loc + kernel[1], width_loc:width_loc + kernel[0]] += (original_score - score[0][label].item()) * factor

        return np.array(score_matrix), image_score_heatmap, image_history

    def visualise_occlusion_sensitivity(self, image, kernel=(5, 5), figsize=(10, 10), fontsize=10, cmap=plt.cm.bwr, factor=255):
        """
        Source: Taken from DEEPLIZARD.com and modified.
        URL: https://deeplizard.com/learn/video/0LhiS6yu2qQ

        Graphs the confusion matrix
        :param image:
        :param kernel:
        :param figsize:
        :param cmap: Colour mapping
        :param factor:

        :return: fig1, fig2; Figure handles for the figures plotted by this method
        """

        matrix, image_matrix, _ = self.occlusion_sensitivity(image, kernel, factor=factor)

        fig1 = plt.figure(figsize=figsize)
        img1 = np.moveaxis(image_matrix, 0, 2) / 255.0
        plt.title("Occlusion Sensitivity Heatmap with Overlay", fontsize=fontsize)

        plt.imshow(img1)

        fig2 = plt.figure(figsize=figsize)
        vmin = np.min(matrix)
        vmax = np.max(matrix)
        vcenter = 0
        if vmin >= vcenter or vmax <= vcenter:
            vcenter = vmin + (vmax - vmin) / 2

        if vmin > 0 and vmax > 0:
            cmap = plt.cm.Reds
        elif vmin < 0 and vmax < 0:
            cmap = plt.cm.Blues

        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
        plt.imshow(matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)

        plt.title("Occlusion Sensitivity Heatmap", fontsize=fontsize)
        plt.colorbar()
        tick_marks = np.arange(len(matrix))
        plt.xticks(tick_marks, range(len(matrix)))
        plt.yticks(tick_marks, range(len(matrix)))

        fmt = '.2f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt), horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black", fontsize=fontsize)

        plt.tight_layout()
        return fig1, fig2

    @staticmethod
    def classes():
        return ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    @staticmethod
    def classes_dictionary():
        return {"angry": 0, "disgusted": 1, "fearful": 2, "happy": 3, "neutral": 4, "sad": 5, "surprised": 6}
