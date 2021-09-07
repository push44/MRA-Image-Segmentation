import torch

from tqdm import tqdm

###################### TRAIN ######################
def train_one_step(model, data, optimizer, device):
    optimizer.zero_grad()

    for k, v in data.items():
        data[k] = v.to(device)

    output, loss = model(**data)
    output = output.cpu().detach().numpy().tolist()

    loss.backward()
    optimizer.step()

    return output, float(loss)

def train_one_epoch(model, data_loader, optimizer, device):
    model.train()

    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        output, loss = train_one_step(model, data, optimizer, device)
        total_loss += loss

    return total_loss/len(data_loader)

###################### VALIDATE ######################
def validate_one_step(model, data, device):

    for k, v in data.items():
        data[k] = v.to(device)

    output, loss = model(**data)
    output = output.cpu().detach().numpy().tolist()

    return output, float(loss)

def validate_one_epoch(model, data_loader, device):
    model.eval()

    total_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            output, loss = validate_one_step(model, data, device)
        total_loss += loss

    return total_loss/len(data_loader)