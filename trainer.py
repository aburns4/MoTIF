import torch
import os
from datetime import datetime
import numpy as np
import os
from sklearn.metrics import average_precision_score, recall_score, accuracy_score, f1_score, precision_score

class Trainer():
    def __init__(self, train_loader, test_loader, model, criterion, optimizer, device,
                 config, bs=126, n_epochs=50, save_model=True, exp_dir='experiments') -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.batch_size = bs
        self.epochs = n_epochs
        self.save_model = save_model
        self.save_dir = exp_dir

    def eval_metrics(self, class_total, class_correct, gt, predictions):
        for i in range(2):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    str(i), 100 * class_correct[i] / class_total[i],
                    np.sum(class_correct[i]), np.sum(class_total[i])))

        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

        flat_preds = [y for x in predictions for y in x]

        ap = average_precision_score(gt, flat_preds)
        p = precision_score(gt, flat_preds, pos_label=0, zero_division=0)
        r = recall_score(gt, flat_preds, pos_label=0, zero_division=0)
        acc = accuracy_score(gt, flat_preds)
        f1 = f1_score(gt, flat_preds, pos_label=0, zero_division=0)
        print('Average Precision: %f' % ap)
        print('Precision: %f' % p)
        print('Recall: %f' % r)
        print('Accuracy: %f' % acc)
        print('F1: %f' % f1)

        metrics = ['Average Precision: ' + str(ap),
                   'Precision: ' + str(p),
                   'Recall: ' + str(r),
                   'Accuracy: ' + str(acc),
                   'F1: ' + str(f1)]
        save_txt = '\n'.join(metrics)
        results_path = self.config[:-5] + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + '.txt'
        save_path = os.path.join(self.save_dir, results_path)
        with open(save_path, 'w') as f:
            f.write(save_txt)

class ComboTrainer(Trainer):
    def __init__(self, train_loader, test_loader, model, criterion, optimizer, device, 
                 config, bs=126, n_epochs=50, save_model=True, exp_dir='experiments') -> None:
        super().__init__(train_loader, test_loader, model, criterion, optimizer, device,
                 config, bs, n_epochs, save_model, exp_dir)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            # monitor training loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            for t_data, v1_data, v2_data, target in  self.train_loader:
                t_data = t_data.to(self.device)
                v1_data = v1_data.to(self.device)
                v2_data = v2_data.to(self.device)
                target = target.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(t_data, v1_data, v2_data)
                # print(output)
                # calculate the loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * t_data.size(0)

            # print training statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(self.train_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch + 1,
                train_loss
            ))

        if self.save_model:
            model_path = 'feasibility_mlp_' + self.config[:-5] + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + '.pt'
            save_path = os.path.join(self.save_dir, model_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_loss,
                        }, save_path)
    
    def eval(self):
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        predictions = []
        gt = []
        self.model.eval()  # prep model for *evaluation*
        i = 0
        for t_data, v1_data, v2_data, target in self.test_loader:
            t_data = t_data.to(self.device)
            v1_data = v1_data.to(self.device)
            v2_data = v2_data.to(self.device)
            target = target.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(t_data, v1_data, v2_data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update test loss
            test_loss += loss.item() * t_data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            predictions.append(pred.cpu())

            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                try:
                    label = target.data[i]
                    gt.append(label.cpu())
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
                except:
                    break
            i += 1

        # calculate and print avg test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        # print and save metrics
        self.eval_metrics(class_total, class_correct, gt, predictions)

class VisTrainer(Trainer):
    def __init__(self, train_loader, test_loader, model, criterion, optimizer, device, 
                 config, bs=126, n_epochs=50, save_model=True, exp_dir='experiments') -> None:
        super().__init__(train_loader, test_loader, model, criterion, optimizer, device,
                                      config, bs, n_epochs, save_model, exp_dir)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            # monitor training loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            for t_data, v_data, target in  self.train_loader:
                t_data = t_data.to(self.device)
                v_data = v_data.to(self.device)
                target = target.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(t_data, v_data)
                # print(output)
                # calculate the loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * t_data.size(0)

            # print training statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(self.train_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch + 1,
                train_loss
            ))

        if self.save_model:
            model_path = 'feasibility_mlp_' + self.config[:-5] + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + '.pt'
            save_path = os.path.join(self.save_dir, model_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_loss,
                        }, save_path)
    
    def eval(self):
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        predictions = []
        gt = []
        self.model.eval()  # prep model for *evaluation*
        i = 0
        for t_data, v_data, target in self.test_loader:
            t_data = t_data.to(self.device)
            v_data = v_data.to(self.device)
            target = target.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(t_data, v_data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update test loss
            test_loss += loss.item() * t_data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            predictions.append(pred.cpu())

            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                try:
                    label = target.data[i]
                    gt.append(label.cpu())
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
                except:
                    break
            i += 1

        # calculate and print avg test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        # print and save metrics
        self.eval_metrics(class_total, class_correct, gt, predictions)

class LangTrainer(Trainer):
    def __init__(self, train_loader, test_loader, model, criterion, optimizer, device, 
                 config, bs=126, n_epochs=50, save_model=True, exp_dir='experiments') -> None:
        super().__init__(train_loader, test_loader, model, criterion, optimizer, device,
                                      config, bs, n_epochs, save_model, exp_dir)


    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            # monitor training loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            for t_data, vt_data, target in  self.train_loader:
                t_data = t_data.to(self.device)
                vt_data = vt_data.to(self.device)
                target = target.to(self.device)

                # clear the gradients of all optimized variables
                self.optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(t_data, vt_data)
                # calculate the loss
                loss = self.criterion(output, target)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                self.optimizer.step()
                # update running training loss
                train_loss += loss.item() * t_data.size(0)

            # print training statistics
            # calculate average loss over an epoch
            train_loss = train_loss / len(self.train_loader.dataset)

            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch + 1,
                train_loss
            ))

        if self.save_model:
            model_path = 'feasibility_mlp_' + self.config[:-5] + '_' + datetime.now().strftime("%m_%d_%Y_%H_%M") + '.pt'
            save_path = os.path.join(self.save_dir, model_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': train_loss,
                        }, save_path)
    
    def eval(self):
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(2))
        class_total = list(0. for i in range(2))
        predictions = []
        gt = []
        self.model.eval()  # prep model for *evaluation*
        i = 0
        for t_data, vt_data, target in self.test_loader:
            t_data = t_data.to(self.device)
            vt_data = vt_data.to(self.device)
            target = target.to(self.device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(t_data, vt_data)
            # calculate the loss
            loss = self.criterion(output, target)
            # update test loss
            test_loss += loss.item() * t_data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            predictions.append(pred.cpu())

            # compare predictions to true label
            correct = np.squeeze(pred.eq(target.data.view_as(pred)))
            # calculate test accuracy for each object class
            for i in range(self.batch_size):
                try:
                    label = target.data[i]
                    gt.append(label.cpu())
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
                except:
                    break
            i += 1

        # calculate and print avg test loss
        test_loss = test_loss / len(self.test_loader.dataset)
        print('Test Loss: {:.6f}\n'.format(test_loss))

        # print and save metrics
        self.eval_metrics(class_total, class_correct, gt, predictions)
