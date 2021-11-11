from numpy import histogram
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

    # Tracking for tensorboard
    bias_c = {
        "convblock1": model_c.convblock1.bias.cpu().detach(),
        "convblock2": model_c.convblock2.bias.cpu().detach(),
        "convblock3": model_c.convblock3.bias.cpu().detach()
    }

    weight_c = {
        "convblock1": model_c.convblock1.weight.cpu().detach(),
        "convblock2": model_c.convblock2.weight.cpu().detach(),
        "convblock3": model_c.convblock3.weight.cpu().detach()
    }

    grad_c = {
        "convblock1": model_c.convblock1.weight.grad.cpu().detach(),
        "convblock2": model_c.convblock2.weight.grad.cpu().detach(),
        "convblock3": model_c.convblock3.weight.grad.cpu().detach()
    }
    
    histogram_critic = {
        "bias": bias_c,
        "weight": weight_c,
        "grad": grad_c
    }

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

    # Tracking for tensorboard
    bias_s = {
        "conv1h": model_s.conv1h.bias.cpu().detach(),
        "conv2h": model_s.conv2h.bias.cpu().detach(),
        "conv3h": model_s.conv3h.bias.cpu().detach(),
        "conv4h": model_s.conv4h.bias.cpu().detach(),

        "conv1l": model_s.conv1l.bias.cpu().detach(),
        "conv2l": model_s.conv2l.bias.cpu().detach(),
        "conv3l": model_s.conv3l.bias.cpu().detach(),
        "conv4l": model_s.conv4l.bias.cpu().detach(),

        "convf1": model_s.convf1.bias.cpu().detach(),
        "convf2": model_s.convf2.bias.cpu().detach(),
        "convf3": model_s.convf3.bias.cpu().detach(),
    }

    weight_s = {
        "conv1h": model_s.conv1h.weight.cpu().detach(),
        "conv2h": model_s.conv2h.weight.cpu().detach(),
        "conv3h": model_s.conv3h.weight.cpu().detach(),
        "conv4h": model_s.conv4h.weight.cpu().detach(),

        "conv1l": model_s.conv1l.weight.cpu().detach(),
        "conv2l": model_s.conv2l.weight.cpu().detach(),
        "conv3l": model_s.conv3l.weight.cpu().detach(),
        "conv4l": model_s.conv4l.weight.cpu().detach(),

        "convf1": model_s.convf1.weight.cpu().detach(),
        "convf2": model_s.convf2.weight.cpu().detach(),
        "convf3": model_s.convf3.weight.cpu().detach()
    }
    
    grad_s = {
        "conv1h": model_s.conv1h.weight.grad.cpu().detach(),
        "conv2h": model_s.conv2h.weight.grad.cpu().detach(),
        "conv3h": model_s.conv3h.weight.grad.cpu().detach(),
        "conv4h": model_s.conv4h.weight.grad.cpu().detach(),

        "conv1l": model_s.conv1l.weight.grad.cpu().detach(),
        "conv2l": model_s.conv2l.weight.grad.cpu().detach(),
        "conv3l": model_s.conv3l.weight.grad.cpu().detach(),
        "conv4l": model_s.conv4l.weight.grad.cpu().detach(),

        "convf1": model_s.convf1.weight.grad.cpu().detach(),
        "convf2": model_s.convf2.weight.grad.cpu().detach(),
        "convf3": model_s.convf3.weight.grad.cpu().detach()
    }

    histogram_segmentor = {
        "bias": bias_s,
        "weight": weight_s,
        "grad": grad_s
    }

    histogram = {
        "segmentor": histogram_segmentor,
        "critic": histogram_critic
    }
    return float(l1_loss_s), float(dice_loss), histogram

def train_one_epoch(model_s, optimizer_s, model_c, optimizer_c, data_loader, device):

    total_l1_loss = 0
    total_dice_loss = 0
    for data in tqdm(data_loader):
        l1_loss, dice_loss, histogram = train_one_step(data, model_s, model_c, optimizer_s, optimizer_c, device)

        total_l1_loss+=l1_loss
        total_dice_loss+=dice_loss

    return total_l1_loss/len(data_loader), total_dice_loss/len(data_loader), histogram

###################### VALIDATE ######################
def validate_one_step(data, model_s, model_c, device):

    for k, v in data.items():
        data[k] = v.to(device)

    image = data["image"]
    del data["image"]

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