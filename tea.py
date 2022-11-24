import torch
from torch import nn, optim
from torch.nn import functional as F
from utils.load_data_Fnt10 import load_data_Fnt10
from utils import evaluate_accuracy
from tqdm import tqdm
from KD.Net import TeacherNet, StudentNet



if __name__ == '__main__':
    INPUT_SIZE = null
    BATCH_SIZE = 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacherNet = TeacherNet(classes=10)
    teacherNet = teacherNet.to(device)
    optimizer = optim.Adam(teacherNet.parameters(), lr=1e-3)
    lossFN = nn.CrossEntropyLoss()

    trainDL, valDL = load_data_Fnt10(INPUT_SIZE, BATCH_SIZE)

    num_epochs = 30
    for epoch in range(num_epochs):
        sum_loss = 0
        sum_acc = 0
        batch_count = 0
        n = 0
        for X, y in tqdm(trainDL):
            X = X.to(device)
            y = y.to(device)
            y_pred = teacherNet(X)

            loss = lossFN(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sum_loss += loss.cpu().item()
            sum_acc += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(valDL, teacherNet)
        print("epoch %d: loss=%.4f \t acc=%.4f \t test acc=%.4f" % (epoch + 1, sum_loss / n, sum_acc / n, test_acc))
    torch.save(teacherNet.state_dict(), './teacherNet.pth')
