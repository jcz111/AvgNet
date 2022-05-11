"""
CNN1D
"""
import pickle

import numpy as np
import csv
import copy
import torch.nn.functional as F
from torchsummary import summary
from tqdm import tqdm
import torch
import our_model
import args


def prepare_data(args):
    # Load data
    if args.dataset == '10a':
        data_path_I = './dataset/10a_Train_I.npy'  # Channel I in the training set
        data_path_Q = './dataset/10a_Train_Q.npy'  # Channel Q in the training set
        train_label_path = './dataset/10a_TrainSnrY.npy'  # Training set labels with SNR
        test_path_I = './dataset/10a_Test_I.npy'  # Channel I in the test set
        test_path_Q = './dataset/10a_Test_Q.npy'  # Channel Q in the test set
        test_label_path = './dataset/10a_TestSnrY.npy'  # Test set labels with SNR
        train_I = np.load(data_path_I)  # [176000, 128]
        train_Q = np.load(data_path_Q)  # [176000, 128]
        train_2 = np.stack([train_I, train_Q], axis=-1)  # [176000, 128, 2]
        train_label = np.load(train_label_path)[1]  # Training set labels
        test_I = np.load(test_path_I)  # [44000, 128]
        test_Q = np.load(test_path_Q)  # [44000, 128]
        test_2 = np.stack([test_I, test_Q], axis=-1)  # [44000, 128, 2]
        test_label = np.load(test_label_path)[1]  # Test set labels
        print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)

        # np->tensor
        train_2 = torch.from_numpy(train_2)  # [312000 , 128, 2]
        train_label = torch.from_numpy(train_label)
        test_2 = torch.from_numpy(test_2)  # [156000 , 128, 2]
        test_label = torch.from_numpy(test_label)
    elif args.dataset == '10b':
        Xd = pickle.load(open('./dataset/RML2016.10b.dat', 'rb'), encoding='iso-8859-1')
        snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
        X = []
        lbl = []
        for mod in mods:
            for snr in snrs:
                X.append(Xd[(mod, snr)])
                for i in range(Xd[(mod, snr)].shape[0]):
                    lbl.append((mod, snr))
        X = np.vstack(X)
        # Split the data 1:1 into train and test sets
        np.random.seed(2016)
        n_examples = X.shape[0]
        n_train = int(n_examples * 0.5)
        train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
        test_idx = list(set(range(0, n_examples)) - set(train_idx))
        train_2 = X[train_idx]  # [600000, 2, 128]
        test_2 = X[test_idx]  # [600000, 2, 128]
        train_label = np.array(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))  # [600000, ]
        test_label = np.array(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))  # [600000, ]

        # np->tensor
        train_2 = torch.from_numpy(train_2).permute(0, 2, 1)  # [600000, 128, 2]
        train_label = torch.from_numpy(train_label)
        test_2 = torch.from_numpy(test_2).permute(0, 2, 1)  # [600000, 128, 2]
        test_label = torch.from_numpy(test_label)
        print(train_2.shape, train_label.shape, test_2.shape, test_label.shape)
    else:
        print('Dataset loading error!!')

    data_sizes = {'train': len(train_label), 'test': len(test_label)}
    train_2 = torch.unsqueeze(train_2, 1)  # [312000, 1, 128, 2]
    test_2 = torch.unsqueeze(test_2, 1)  # [156000, 1, 128, 2]
    train_signal = torch.utils.data.TensorDataset(train_2, train_label)
    test_signal = torch.utils.data.TensorDataset(test_2, test_label)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_signal,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_signal,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers)
    # Here, we load all the data onto the GPU first
    trains = []
    for data1, label1 in train_loader:
        data1 = data1.cuda().float()  # [batch_size, 1, 128, 2]
        label1 = torch.unsqueeze(label1, 1)
        label1 = label1.cuda().long()  # [batch_size, 1]
        trains.append((data1, label1))
    tests = []
    for data1, label1 in test_loader:
        data1 = data1.cuda().float()  # [batch_size, 1, 128, 2]
        label1 = torch.unsqueeze(label1, 1)
        label1 = label1.cuda().long()  # [batch_size, 1]
        tests.append((data1, label1))

    return trains, tests, data_sizes


def train(model, trains, optimizer, scheduler, data_sizes):
    model.train()
    loss_all = 0
    correct = 0
    pbar = tqdm(trains)
    for batch_data, batch_label in pbar:
        optimizer.zero_grad()
        output, _, _ = model(batch_data)
        loss = F.nll_loss(output, batch_label.view(-1))
        loss.backward()
        now_loss = batch_label.size(0) * loss.item()
        loss_all += now_loss
        pred = output.max(dim=1)[1]
        now_correct = pred.eq(batch_label.view(-1)).sum().item()
        correct += now_correct
        optimizer.step()
        epoch_loss = now_loss / batch_label.size(0)
        epoch_acc = now_correct / batch_label.size(0) * 1.0
        pbar.set_postfix({'Set': 'Train', 'Epoch Loss': '{:.4f}'.format(epoch_loss),
                          'Epoch Acc': '{:.4f}'.format(epoch_acc)})
    scheduler.step()
    return loss_all / data_sizes['train'], correct / data_sizes['train']


@torch.no_grad()
def test(model, tests, data_sizes):
    model.eval()
    loss_all = 0
    correct = 0
    signal_num = 0
    pbar = tqdm(tests)
    for batch_data, batch_label in pbar:
        output, _, _ = model(batch_data)
        loss = F.nll_loss(output, batch_label.view(-1))
        loss_all += batch_label.size(0) * loss.item()
        pred = output.max(dim=1)[1]
        correct += pred.eq(batch_label.view(-1)).sum().item()
        signal_num += batch_label.size(0)
        epoch_loss = loss_all / signal_num
        epoch_acc = correct / signal_num * 1.0
        pbar.set_postfix({'Set': 'Test', 'Average Loss': '{:.4f}'.format(epoch_loss),
                          'Average Acc': '{:.4f}'.format(epoch_acc)})
    return loss_all / data_sizes['test'], correct / data_sizes['test']


def main():
    prog_args = args.arg_parse()
    #  Load dataset
    trains, tests, data_sizes = prepare_data(prog_args)
    #  Initialize the model
    model = our_model.Net().cuda()
    summary(model, (1, 128, 2))
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print("============================")
    optimizer = torch.optim.Adam(model.parameters(), lr=prog_args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    with open("./result/record/result_lr={}_filter={}.csv".format(prog_args.lr, prog_args.num_filter), 'a', newline='') as t1:
        writer_train1 = csv.writer(t1)
        writer_train1.writerow(['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'best_test_acc'])
    best_test_acc = test_acc = 0
    for epoch in range(1, prog_args.epochs):
        # scheduler.step()
        train_loss, train_acc = train(model, trains, optimizer, scheduler, data_sizes)
        test_loss, test_acc = test(model, tests, data_sizes)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, './result/model/best_model_parameters_lr={}_filter={}.pth'.format(prog_args.lr, prog_args.num_filter))
        print('Epoch: {:03d}, Train Loss: {:.7f}, Train Acc: {:.7f}, Test Loss: {:.7f},'
              'Test Acc: {:.7f}, Best test Acc: {:.7f}'.format
              (epoch, train_loss, train_acc, test_loss, test_acc, best_test_acc))
        print("======================================================================")
        with open("./result/record/result_lr={}_filter={}.csv".format(prog_args.lr, prog_args.num_filter), 'a', newline='') as t1:
            writer_train1 = csv.writer(t1)
            writer_train1.writerow([epoch, train_loss, train_acc, test_loss, test_acc, best_test_acc])


if __name__ == '__main__':
    main()
