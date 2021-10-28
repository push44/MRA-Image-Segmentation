import torch
from tqdm import tqdm

###################### TRAIN ######################
def train_one_step(data, model_s, model_c, optimizer_s, optimizer_c, device):
    ######################## TRAIN CRITIC ########################
    model_c.zero_grad()

    for k, v in data.items():
        data[k] = v.to(device)

    image = data["image"]
    del data["image"]

    with torch.no_grad():

        output, _ = model_s(**data)
        output = output.cpu().detach()

    prediction_masked_image = output.to(device) * image.clone()
    gt_masked_image = data["mask"] * image.clone()
    data_c = {
        "prediction_masked_image": prediction_masked_image.to(device),
        "gt_masked_image": gt_masked_image.to(device) 
    }
    _, l1_loss_c = model_c(**data_c)
    l1_loss_c.backward(retain_graph = True)
    optimizer_c.step()

    ######################## TRAIN SEGMENTOR ########################
    model_s.zero_grad()

    output, dice_loss = model_s(**data)
    prediction_masked_image = output.to(device) * image.clone()
    gt_masked_image = data["mask"] * image.clone()
    data_c = {
        "prediction_masked_image": prediction_masked_image.to(device),
        "gt_masked_image": gt_masked_image.to(device) 
    }
    _, l1_loss_s = model_c(**data_c)
    l1_loss_s.backward()
    optimizer_s.step()

    return l1_loss_s, dice_loss

def train_one_epoch(model_s, optimizer_s, model_c, optimizer_c, data_loader, device):

    total_l1_loss = 0
    total_dice_loss = 0
    for data in tqdm(data_loader):
        l1_loss, dice_loss = train_one_step(data, model_s, model_c, optimizer_s, optimizer_c, device)

        total_l1_loss+=l1_loss
        total_dice_loss+=dice_loss

    return total_l1_loss/len(data_loader), total_dice_loss/len(data_loader)

###################### VALIDATE ######################
def validate_one_step(data, model_s, model_c, device):

    image = data["image"]
    del data["image"]

    for k, v in data.items():
        data[k] = v.to(device)

    output, _ = model_s(**data)
    output = output.cpu().detach()

    prediction_masked_image = output.to(device) * image.clone()
    gt_masked_image = data["mask"] * image.clone()
    data_c = {
        "prediction_masked_image": prediction_masked_image.to(device),
        "gt_masked_image": gt_masked_image.to(device) 
    }
    _, l1_loss = model_c(**data_c)

    return float(l1_loss)

def validate_one_epoch(model_s, model_c, data_loader, device):
    model_s.eval()
    model_c.eval()

    total_l1_loss = 0
    for data in tqdm(data_loader):
        with torch.no_grad():
            l1_loss = validate_one_step(data, model_s, model_c, device)
        total_l1_loss+=l1_loss

    return total_l1_loss/len(data_loader)

###################### PREDICT ######################
def predict_one_step(model, data, device):
    for k, v in data.items():
        data[k] = v.to(device)

    output, _ = model(**data)
    output = output.cpu().detach().numpy().tolist()

    return output

def predict_one_epoch(model, data_loader, device):
    model.eval()

    print("Predicting...")
    fnames = []
    preds = []
    for data in tqdm(data_loader):
        fnames.extend(data["filename"])
        del data["filename"]
        with torch.no_grad():
            output = predict_one_step(model, data, device)
        preds.extend(output)

    return preds, fnames