import csv
import matplotlib.pyplot as plt
import os

class TrainingInformation():
    """
    Used during the training of the Neural Network to store information. 
    """
    def write_hp_csv(self, dir, n_best_hp, monitor_metric):
        fieldnames = ['model_rank', 'lr', 'batch_size', 'metric_loss']
        with open(f'{dir}/hp_search_results.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for i, trial in enumerate(n_best_hp):
                hps = trial.hyperparameters
                validation_loss = trial.metrics.get_best_value(monitor_metric)
                hps.get('lr')
                row_dict = {
                    'model_rank': i+1, 
                    'lr': hps.get('lr'),
                    'batch_size': hps.get('batch_size'),
                    'metric_loss': validation_loss
                }
                writer.writerow(row_dict)

    def plot_graph(self, history, name, directory, monitor_metric):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss)+1)
        plt.plot(epochs, loss, 'blue', label='Training_loss')
        plt.plot(epochs, val_loss, 'orange', label='validation_loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        

        dir = directory + '/training'
        if not os.path.exists(dir):
            os.makedirs(dir)
        plt.savefig(f'{dir}/train_val_loss_{name}.png')
        plt.close()

        acc = history.history[monitor_metric]
        val_acc = history.history[f'val_{monitor_metric}']
        plt.plot(epochs, acc, 'blue', label='Training accuracy')
        plt.plot(epochs, val_acc, 'orange', label='validation accuracy')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{dir}/train_val_acc_{name}.png')
        plt.close()
