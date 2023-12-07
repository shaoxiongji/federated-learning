import xlsxwriter
from torch.utils.data import DataLoader
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

from models.Nets import CNNimu
from utils.options import args_parser



class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor[:, :-1].reshape(-1,1,9,100)
        self.targets = data_tensor[:, -1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    batch_loss = []
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        batch_loss.append( F.cross_entropy(log_probs, target.long()).item() )
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss = sum(batch_loss)/len(batch_loss)
    print('Test set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss

if __name__ == '__main__':

    torch.cuda.empty_cache
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    #Load dataset
    dataset = torch.load('IMU_data.pt')
    dataset = dataset.float()
    print(dataset.shape)

    dataset = CustomDataset(dataset)

    #percentages are arbitrary
    total_count = len(dataset)
    print("Total Count: ", total_count)
    train_count = int(0.8 * total_count)  # 80% for training
    test_count = int(0.1 * total_count)   # 10% for testing
    validation = total_count - train_count - test_count # 10% for validation
    dataset_train, dataset_test, dataset_validation = random_split(dataset, [train_count, test_count, validation])

    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
    validation_loader = DataLoader(dataset_validation, batch_size=32, shuffle=False)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)


    #CNN 
    #net_glob = CNNimu(args=args)

    #mnasnet1_3
    #net_glob = torchvision.models.mnasnet1_3()
    #net_glob.layers[0] = torch.nn.Conv2d(1, 40, 3, padding=1, stride=2, bias=False) #40 for mnasnet1_3 and 16 for 0_5
    #net_glob.classifier[1] = torch.nn.Linear(in_features=1280, out_features=10,bias=True)

    #Effic
    #net_glob = torchvision.models.efficientnet_b0()
    #net_glob.features[0] = torchvision.ops.Conv2dNormActivation(1, 32, kernel_size=(3, 3), stride=(2, 2), norm_layer=torch.nn.BatchNorm2d, activation_layer=torch.nn.SiLU)
    #net_glob.classifier = torch.nn.Sequential(torch.nn.Dropout(p=0.2, inplace=True),torch.nn.Linear(in_features=1280, out_features=10))

    #ResNet
    net_glob = torchvision.models.resnet18()
    net_glob.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    net_glob.fc = torch.nn.Linear(in_features=512, out_features=3, bias=True)

    net_glob.to(args.device)
    #Update workbook names based off model

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)

    list_loss = []
    list_test_acc = []
    list_test_loss = []
    list_time = []
    epochs = 100
    net_glob.train()
    for epoch in range(epochs):
        net_glob.train()
        time1 = time.time()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target.long())
            loss.backward()
            optimizer.step()
            if batch_idx % 50 == 0: #{:.0f}
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime Take: '.format(
                    epoch, batch_idx * len(data),len(train_loader.dataset),
                            100.*batch_idx/len(train_loader), loss.item()) + str(time.time()-time1))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)

        print('Train loss:', loss_avg)
        list_loss.append(loss_avg)

        timePerEpoch = time.time()-time1
        print("Time taken: ", timePerEpoch,'\n')
        list_time.append(timePerEpoch)


        print("Training accuracy for epoch{}:".format(epoch))
        train_acc, train_loss = test(net_glob, train_loader)
        #list_test_acc.append(train_acc)
       # trainAccTime=time.time()-timePerEpoch-time1
        #print("Time for testing: ",trainAccTime,'\n')
        #list_train_loss = []
        #list_train_acc = []

        #if(epoch<49):
        print("\nTesting accuracy for epoch{}:".format(epoch))
        test_acc, test_loss = test(net_glob, test_loader)
        list_test_acc.append(test_acc)
        list_test_loss.append(test_loss)
        testAccTime=time.time()-timePerEpoch-time1
        print("Time for testing: ",testAccTime,'\n')

        #elif( ( (epoch+1) %10) ==0):
            #print("Testing accuracy for epoch{}:".format(epoch))
            #test_acc, test_loss = test(net_glob, test_loader)
            #list_test_acc.append(test_acc)
        #elif(epoch==99):
            #print("Testing accuracy for epoch{}:".format(epoch))
            #test_acc, test_loss = test(net_glob, test_loader)
            #list_test_acc.append(test_acc)


    #Saves Test Accuracy Per Epoch
    workbook = xlsxwriter.Workbook('RESNET_TestAccuracyFor{}_{}Epoch.xlsx'.format(args.model,epochs))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for acc in (list_test_acc):
        worksheet.write(row,col,acc)
        row+=1
    workbook.close()

    #Saves Test Loss Per Epoch
    workbook = xlsxwriter.Workbook('RESNET_TestLossFor{}_{}Epoch.xlsx'.format(args.model,epochs))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for l in (list_test_loss):
        worksheet.write(row,col,l)
        row+=1
    workbook.close()

    #Saves Loss Per Epoch
    #workbook = xlsxwriter.Workbook('RESNET__LossAvgOver{}Epochs.xlsx'.format(epochs))
    #worksheet = workbook.add_worksheet()
    #row = 0
    #col = 0
    #for los in (list_loss):
    #    worksheet.write(row,col,los)
    #    row+=1
    #workbook.close()

    #Saves time Per Epoch
    workbook = xlsxwriter.Workbook('RESNET_TimePer{}Epochs.xlsx'.format(epochs))
    worksheet = workbook.add_worksheet()
    row = 0
    col = 0
    for t in (list_time):
        worksheet.write(row,col,t)
        row+=1
    workbook.close()

    #evaluating
    print('\nTest on', len(dataset_validation), 'samples')
    test_acc, test_loss = test(net_glob, validation_loader)
