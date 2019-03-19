from transformations.transformations import ToTensor
from utils.utils import read_config, Counters, save_model, train_test_split
from torchvision import transforms
import torch
from torch import optim
from torch import nn
import torch.utils.data as data_utils
from dnn.network import DNN
import logging
from dataset.dataset_class import CustomDataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def train(train_dataloader, model, optimizer, criterion, epochs=10):
    model.train()
    counter = Counters()
    loss = None
    for epoch in range(1, epochs + 1):
        for j, sample in enumerate(train_dataloader):
            logger.info('Batch number {0}'.format(j))
            images = sample['image'].to(device)
            labels = sample['labels'].to(device)

            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counter.update(loss)
        logger.info('Epoch {}, loss {}'.format(epoch, counter.loss.item()))
    return loss


def test(test_dataset, model, criterion):
    model.eval()
    counter = Counters()
    loss = None
    with torch.no_grad():
        for j, sample in enumerate(test_dataset):
            logger.info('Batch number {0}'.format(j))
            images = sample['image'].to(device)
            labels = sample['labels'].to(device)
            outputs = model.forward(images)
            loss = criterion(outputs, labels)
            counter.update(loss)
    logger.info('Test loss {}'.format(counter.loss.item()))
    return loss


def main():
    train_epoch = 0
    config = read_config('defaults.cfg')

    if torch.cuda.is_available():
        logger.info('device: GPU')
    else:
        logger.info('device: CPU')

    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    total_epochs = config['total_epochs']
    batch_epochs = config['batch_epochs']

    model = DNN().to(device)

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_transforms = transforms.Compose([ToTensor()])
    train_dataset = CustomDataset(data_dir=config['data'],
                                  labels_file=config['labels'],
                                  transforms=train_transforms
                                  )

    test_dataset = CustomDataset(data_dir=config['data'],
                                 labels_file=config['labels'],
                                 transforms=train_transforms
                                 )

    data_len = len(train_dataset)

    train_idx, test_idx = train_test_split(data_len, train_size=config['train_size'])

    train_sampler = data_utils.RandomSampler(train_idx)
    test_sampler = data_utils.RandomSampler(test_idx)

    train_dataloader = data_utils.DataLoader(train_dataset,
                                             sampler=train_sampler,
                                             batch_size=batch_size,
                                             shuffle=False)

    test_dataloader = data_utils.DataLoader(test_dataset,
                                            sampler=test_sampler,
                                            batch_size=batch_size,
                                            shuffle=False)

    while train_epoch < total_epochs:
        train_loss = train(train_dataloader, model, optimizer, criterion, epochs=batch_epochs)
        train_epoch += batch_epochs
        test_loss = test(test_dataloader, model, criterion)
        save_model(model, optimizer, train_loss, test_loss, train_epoch)


if __name__ == '__main__':
    main()
