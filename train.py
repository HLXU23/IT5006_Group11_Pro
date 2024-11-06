import torch
import numpy as np

class Trainer:
    def __init__(self, model, model_optimizer, print_every, epochs=200, device='cpu', result_name='RUL_pred', max_rul = 125):
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss()
        self.result_name = result_name
        self.log_file = f'./log/{result_name}.log'
        self.max_rul = max_rul

    def write_train_log(self, log_message):
        print(log_message)
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")

    def train_single_epoch(self, dataloader):
        running_loss = 0

        length = len(dataloader)

        for batch_index, data in enumerate(dataloader, 0):
            inputs, handcrafted_feature, labels = data
            inputs, handcrafted_feature, labels = inputs.to(self.device), handcrafted_feature.to(self.device), labels.to(self.device)
            self.model_optimizer.zero_grad()
            predictions = self.model(inputs, handcrafted_feature)
            loss = self.criterion(predictions, labels)
            running_loss += loss.item()
            loss.backward()

            self.model_optimizer.step()

            if (batch_index + 1) % self.print_every == 0:
                log_message = 'batch:{}/{}, loss(avg. on {} batches: {}'.format(
                                                                        batch_index + 1,
                                                                        length,
                                                                        self.print_every,
                                                                        running_loss / self.print_every,
                                                                        )
                self.write_train_log(log_message)
                running_loss = 0

    def train(self, train_loader, test_loader, iteration):
        for epoch in range(self.epochs):
            log_message = 'Epoch: {}'.format(epoch + 1)
            self.write_train_log(log_message)
            self.model.train()
            self.train_single_epoch(train_loader)
            current_score, current_RMSE = self.test(test_loader)
            if epoch == 0:
                best_score = current_score
                best_RMSE = current_RMSE
            else:
                if current_score < best_score:
                    best_score = current_score
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_score')
                if current_RMSE < best_RMSE:
                    best_RMSE = current_RMSE
                    self.save_checkpoints(iteration + 1, epoch + 1, 'best_RMSE')
        return float(best_score), float(best_RMSE)

    def save_checkpoints(self, iteration, epoch, which_type):
        state = {
            'iter': iteration,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.model_optimizer.state_dict()
        }
        torch.save(state, f'./checkpoints/{self.result_name}_iteration{iteration}_{which_type}.pth.tar')
        log_message = '{}_checkpoints saved successfully!'.format(which_type)
        self.write_train_log(log_message)

    @staticmethod
    def score(y_true, y_pred):
        score = 0
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        for i in range(len(y_pred)):
            if y_true[i] <= y_pred[i]:
                score = score + np.exp(-(y_true[i] - y_pred[i]) / 10.0) - 1
            else:
                score = score + np.exp((y_true[i] - y_pred[i]) / 13.0) - 1
        return score

    def test(self, test_loader):
        score = 0
        loss = 0
        self.model.eval()
        criterion = torch.nn.MSELoss()
        for batch_index, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, handcrafted_feature, labels = data
                inputs, handcrafted_feature, labels = inputs.to(self.device), handcrafted_feature.to(self.device), labels.to(self.device)
                predictions = self.model(inputs, handcrafted_feature)

                score += self.score(labels * self.max_rul, predictions * self.max_rul)
                loss += criterion(labels * self.max_rul, predictions * self.max_rul) * len(labels)
        loss = (loss / len(test_loader.dataset)) ** 0.5
        log_message = 'test result: score: {}, RMSE: {}'.format(score.item(), loss)
        self.write_train_log(log_message)
        return score.item(), loss
