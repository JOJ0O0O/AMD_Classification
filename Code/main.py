import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from CNN import CNN
from pathlib import Path


def load_images(
    folder: str | Path = "images",
    batch_size: int = 64,
    image_size: tuple[int, int] = (224, 224),
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list[str]]:
    """
    Load the AMDNet23 training and validation image splits.

    Date: 2026-11-19.
    Written by: AI assistant using Copilot SDK in VS Code.
    Add this reusable image-loading function to the AMD
    classification project.
    This corresponds to Level 3 usage of AI, as I tested and validated this function myself

    Args:
        folder: Directory containing the image dataset. Relative paths are
            resolved from the project root.
        batch_size: Number of images in each data-loader batch.
        image_size: Width and height to which each image is resized.
        num_workers: Number of worker processes used to load data.

    Returns:
        A tuple containing the training loader, validation loader, and class
        names in dataset order.

    Raises:
        ValueError: If the folder or one of the required splits is missing,
            or if the splits contain incompatible class definitions.
    """
    # Resolve relative folders consistently, regardless of the current working
    # directory used to start the program.
    project_root = Path(__file__).resolve().parent.parent
    images_folder = (
        (project_root / folder).resolve()
        if not Path(folder).is_absolute()
        else Path(folder).resolve()
    )

    if not images_folder.is_dir():
        raise ValueError(f"Images folder does not exist: {images_folder}")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")
    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError("image_size must contain two positive dimensions")

    # Locate the AMDNet23 dataset below the images folder. This supports its
    # nested storage layout while allowing a direct train/valid layout too.
    split_parents = [
        path.parent
        for path in images_folder.rglob("train")
        if path.is_dir()
        and (path.parent / "valid").is_dir()
    ]

    if not split_parents:
        raise ValueError(
            f"No dataset with 'train' and 'valid' folders was found under "
            f"{images_folder}"
        )

    dataset_root = split_parents[0]
    train_folder = dataset_root / "train"
    valid_folder = dataset_root / "valid"

    # Apply the same deterministic geometric and tensor transforms to every
    # image in both dataset splits.
    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )

    # ImageFolder infers class labels from subdirectory names.
    train_dataset = datasets.ImageFolder(train_folder, transform=transform)
    valid_dataset = datasets.ImageFolder(valid_folder, transform=transform)

    if not train_dataset.samples:
        raise ValueError(f"No training images found in {train_folder}")
    if not valid_dataset.samples:
        raise ValueError(f"No validation images found in {valid_folder}")
    if train_dataset.classes != valid_dataset.classes:
        raise ValueError(
            "Training and validation splits contain different class names"
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, valid_loader, train_dataset.classes

def CNN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean and std
        ])

    train_loader, test_loader, class_names = load_images(folder="images", batch_size=64, image_size=(256, 256), num_workers=0)

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss() # applies softmax internally

    """
    The Code below was directly taken from
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(10):
        model.train()
        running = 0.0
        print(f"epoch {epoch+1}:")
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward() # Eqs. (10.dconvW)-(10.dmaxpool), by autograd
            optimizer.step()
            running += loss.item()
        print(f"epoch {epoch+1}: loss {running/len(train_loader):.4f}")

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    print(f"test accuracy {100*correct/total:.2f}%")


if __name__ == "__main__":
    CNN()
