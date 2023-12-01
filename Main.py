import torch
from tqdm import tqdm
from model.Network import Network
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from DataLoader import get_loader
from Utils import set_seed, TripletLoss
from Opt import parse_args


def main():
    args = parse_args()
    device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")
    if args.data_type == "utk":
        label = torch.Tensor([i for i in range(21, 61)]).to(device)
        num_classes = 40
    elif args.data_type == "cacd":
        label = torch.Tensor([i for i in range(14, 63)]).to(device)
        num_classes = 49
    else:
        raise ValueError("No data type argument.")

    model = Network(num_classes=num_classes)
    model = torch.nn.DataParallel(model).to(device)
    triplet_loss = TripletLoss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=1e-1)

    if args.phase == "train":
        train_loader = get_loader("train", args)
        val_loader = get_loader("val", args)
        train(args.epochs, model, train_loader, device, label, criterion, triplet_loss, optimizer, scheduler,
              val_loader, args.checkpoint_path, args.seed)
    elif args.phase == "test":
        test_loader = get_loader("test", args)
        test(test_loader, model, args.model_path, criterion, device, label)
    else:
        raise ValueError("No phase argument.")


def train(epoch_num, model, train_loader, device, label, criterion, triplet_loss, optimizer, scheduler, val_loader,
          model_path, seed):
    min_train_loss = np.inf
    set_seed(seed)
    for epoch in range(epoch_num):
        print(f"\nEpoch {epoch + 1}/{epoch_num}")
        model.train()
        loss_l1 = 0
        loss_triplet = 0
        print("\nTRAINING:")
        for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            input, positive, negative, gt_ages = data

            anchor, positive, negative, predictions = model(input=input.to(device), positive=positive.to(device),
                                                            negative=negative.to(device), phase="train")
            pred_ages = torch.sum(predictions * label, dim=1)

            mae = criterion(pred_ages, gt_ages.to(device))
            loss_l1 += mae.item()

            triplet = triplet_loss(anchor, positive, negative)
            loss_triplet += triplet.item()

            loss_total = mae + triplet

            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f'MAE: {round(loss_l1 / len(train_loader), 4)}')
        print(f'Triplet: {round(loss_triplet / len(train_loader), 4)}')
        scheduler.step()

        print("\nVALIDATION:")
        model.eval()
        loss_l1 = 0

        with torch.no_grad():
            for i, data in tqdm(enumerate(val_loader), total=len(val_loader)):
                input, gt_ages = data

                predictions = model(input=input.to(device))
                pred_ages = torch.sum(predictions * label, dim=1)

                mae = criterion(pred_ages, gt_ages.to(device))
                loss_l1 += mae.item()

            print(f'MAE: {round(loss_l1 / len(val_loader), 4)}')

            # SAVE MODEL
            if loss_l1 < min_train_loss:
                min_train_loss = loss_l1
                torch.save(model.state_dict(), model_path)


def test(test_loader, model, model_path, criterion, device, label):
    model.load_state_dict(torch.load(model_path))
    print("\nTESTING:")
    loss_l1 = 0
    model.eval()
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
            input, gt_ages = data

            predictions = model(input=input.to(device))
            pred_ages = torch.sum(predictions * label, dim=1)

            mae = criterion(pred_ages, gt_ages.to(device))
            loss_l1 += mae.item()

        print(f'MAE: {round(loss_l1 / len(test_loader), 4)}')


if __name__ == '__main__':
    main()
