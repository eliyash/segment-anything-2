import json
from datetime import datetime
from pathlib import Path

from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os


def main():
    is_windows = os.name == 'nt'

    root_dir = Path(r'C:\Workspace\ChimpanzeesThesis\faces_images') if is_windows else Path(r'/home/ubuntu/faces_work')
    data_dir = root_dir / 'individual_faces_dataset'
    out_dir = root_dir / 'training'

    batch_size = 32
    epochs = 100
    workers = 0 if is_windows else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    train_name = datetime.now().strftime('train__%Y%m%d_%H%M%S')

    model_folder = out_dir / train_name
    model_folder.mkdir(parents=True, exist_ok=True)

    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        transforms.Resize((512, 512)),
        fixed_image_standardization
    ])
    dataset = datasets.ImageFolder(data_dir.as_posix(), transform=trans)
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=len(dataset.class_to_idx)
    ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]

    train_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_inds)
    )
    val_loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )
        info = {'epoch': epoch}
        (model_folder / 'info.json').write_text(json.dumps(info, indent=4))
        torch.save(resnet.state_dict(), model_folder / 'model.pt')

    writer.close()


if __name__ == '__main__':
    main()
