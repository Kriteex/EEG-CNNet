import torch
from torch import nn
from metrics_and_plots import accuracy
from data_preprocess import load_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_losses = []
test_losses = []

def train(model, train_data, train_labels, test_data, test_labels, loss_fn, optimizer, epochs=100):
    train_data_torch = torch.from_numpy(train_data).float().to(device)
    train_labels_torch = torch.from_numpy(train_labels).long().to(device)

    test_data_torch = torch.from_numpy(test_data).float().to(device)
    test_labels_torch = torch.from_numpy(test_labels).long().to(device)
    
    for epoch in range(epochs):

        # Training phase
        model.train()

        # Forward pass
        pred = model(train_data_torch)

        loss = loss_fn(pred, train_labels_torch)
        acc = accuracy(pred, train_labels_torch)

        train_losses.append(loss.cpu().detach().numpy())

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            test_pred = model(test_data_torch)
            test_loss = loss_fn(test_pred, test_labels_torch)
            test_losses.append(test_loss.cpu().detach().numpy())
            test_acc = accuracy(test_pred, test_labels_torch)


        if (epoch) % 10 == 0:
            print(f'Epoch {epoch}/{epochs}, Training Loss: {str(loss.item())[:4]}, Validation Loss: {str(test_loss.item())[:4]}')
            print(f'Training accuracy: {str(acc.item())[:4]}, Test_accuracy: {str(test_acc.item())[:4]}')



def test_all(path, save_dir, model):
    tot_acc = 0
    
    loss_fn_class = nn.CrossEntropyLoss()

    all_labels = []  # list to store all true labels
    all_preds = []  # list to store all predictions

    # Evaluation phase
    model.eval()

    for i in range(45):
        _,_,test_data, test_label = load_data(path, i)



        #==========================CLASS===================================
        test_data_torch = torch.from_numpy(test_data).float().to(device)
        test_labels_torch = torch.from_numpy(test_label).long().to(device)

    
        with torch.no_grad():
            test_class_pred = model(test_data_torch)

            all_labels.append(test_labels_torch)  # append true labels
            all_preds.append(test_class_pred)  # append predictions

            
            test_loss_class = loss_fn_class(test_class_pred, test_labels_torch)
            
            test_acc_class = accuracy(test_class_pred, test_labels_torch)
            

            if tot_acc == 0:
                tot_acc = test_acc_class
            else:
                tot_acc = (tot_acc + test_acc_class)/2

    print("TOTAL TEST ACC: " + str(tot_acc.item())[:4])