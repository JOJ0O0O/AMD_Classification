import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from CNN import CNN
from logistic_regression import LogisticRegression
from pathlib import Path
from skorch import NeuralNetClassifier
from skorch.callbacks import Callback
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, recall_score, f1_score
import numpy as np
import tqdm
import sys
from contextlib import redirect_stdout
from cross_val_config import n_folds, image_size, param_grid #Import Configuration from separate config file

# Optional separate grid for logistic regression. If cross_val_config.py does
# not define logreg_param_grid, the CNN grid is used without its CNN-specific
# "module__..." entries (see without_module_params).
try:
    from cross_val_config import logreg_param_grid
except ImportError:
    logreg_param_grid = None

torch.set_num_threads(16)

def print_epoch_log(text):
    """
    Print so the text of the progress does not get put into results .txt
    """
    print(text, file=sys.stderr)

class FitProgress(Callback):
    """Update the shared progress bar after each model fit.

    Date: 2026-11-22.
    Written by: AI assistant using Copilot SDK in VS Code.
    Add this reusable indexed dataset class to the AMD classification project for the cross validation.
    This corresponds to Level 2 usage of AI, as I tested, changed and validated this function myself.
    """

    progress_bar = None

    def on_train_end(self, net, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.update(1)

class IndexedImageDataset(torch.utils.data.Dataset):
    """
    Load ImageFolder samples lazily from sklearn-provided row indices.

    Date: 2026-11-22.
    Written by: AI assistant using Copilot SDK in VS Code.
    Add this reusable indexed dataset class to the AMD classification project for the cross validation.
    This corresponds to Level 3 usage of AI, as I tested and validated this function myself
    """
    def __init__(self, indices, y=None, image_folder=None):
        self.images = None
        self.indices = None
        if isinstance(indices, (np.ndarray, torch.Tensor)) and indices.ndim > 1:
            self.images = indices
        elif image_folder is None:
            raise ValueError("image_folder must be provided")
        else:
            self.indices = indices
        self.image_folder = image_folder

    def __len__(self) -> int:
        return len(self.images if self.images is not None else self.indices)

    def __getitem__(self, index):
        if self.images is not None:
            return self.images[index], 0
        return self.image_folder[int(self.indices[index])]

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

def without_module_params(grid):
    """
    Return a copy of a parameter grid without "module__..." entries, i.e.
    without constructor arguments of the CNN (e.g. module__channels) that the
    logistic regression does not accept. Works for a dict or a list of dicts.

    Date: 2026-11-25.
    Written by: Claude Opus 5.5 (Anthropic).
    Level: TODO - set after testing and validating it yourself.
    """
    if isinstance(grid, dict):
        return {key: value for key, value in grid.items() if not key.startswith("module__")}
    return [without_module_params(sub_grid) for sub_grid in grid]

def build_module(model_name, train_loader, class_names):
    """
    Return the skorch module, its constructor arguments and the parameter grid
    for the selected model ("cnn" or "logreg").

    Date: 2026-11-25.
    Written by: Claude Opus 5.5 (Anthropic).
    Level: TODO - set after testing and validating it yourself.
    """
    if model_name == "cnn":
        return CNN, {}, param_grid
    if model_name == "logreg":
        # Number of input features = C * H * W of one preprocessed image.
        n_inputs = train_loader.dataset[0][0].numel()
        module_kwargs = {
            "module__n_inputs": n_inputs,
            "module__n_outputs": len(class_names),
        }
        grid = logreg_param_grid if logreg_param_grid is not None else without_module_params(param_grid)
        return LogisticRegression, module_kwargs, grid
    raise ValueError(f"Unknown model: {model_name!r} (use 'cnn' or 'logreg')")

def CNN_test(model_name="cnn"):
    """
    Function to do a Cross validation of the selected model ("cnn" or "logreg")
    for the AMDNet23 dataset.
    The parameters which are validated are set in cross_val_config.py

    Date: 2026-11-25.
    Modified by: Claude Opus 5.5 (Anthropic). Added model_name so the same
    cross validation runs for the CNN and the logistic regression.
    Level: TODO - set after testing and validating the changes yourself.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate image loader
    train_loader, test_loader, class_names = load_images(folder="images", batch_size=64, image_size=image_size, num_workers=0)

    #Select model, its constructor arguments and the parameter grid
    module, module_kwargs, model_param_grid = build_module(model_name, train_loader, class_names)

    #Set up Scorch NeuralNetClassifier with the selected model
    CNN_Classifier = NeuralNetClassifier(
        module = module,
        **module_kwargs,
        criterion = nn.CrossEntropyLoss(),
        max_epochs = 2,
        lr = 0.001,
        batch_size = 64,
        iterator_train__shuffle = True,
        optimizer = optim.Adam,
        device = device,
        train_split = None,
        dataset=IndexedImageDataset,
        dataset__image_folder=train_loader.dataset,
        callbacks=[FitProgress()],
        callbacks__print_log__sink=print_epoch_log)

    #CV Setup
    labels = np.asarray(train_loader.dataset.targets)
    cv_indices = np.arange(len(train_loader.dataset))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=2026)

    #Gridsearch with parameters specified in cross_val_config.py
    grid_search = GridSearchCV(
        CNN_Classifier,
        param_grid=model_param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=1,
        error_score='raise',
    )

    #Fit all the models with progress bar
    """
    Date: 2026-11-22.
    Written by: AI assistant using Copilot SDK in VS Code.
    The below code was adjusted using AI as I was unsure of the tqdm implementation.
    This corresponds to Level 2 usage of AI, as I tested, changed and validated this function myself.
    """
    total_fits = len(ParameterGrid(model_param_grid)) * cv.get_n_splits() +1
    with tqdm.tqdm(
        total=total_fits,
        desc=f"Grid search ({model_name})",
        unit="fit",
        dynamic_ncols=True,
    ) as progress:
        FitProgress.progress_bar = progress
        try:
            grid_search.fit(cv_indices, labels)
        finally:
            FitProgress.progress_bar = None

    #Select best model and best parameters
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    #Predict on test set and count number of correct predictions
    correct = total = 0
    predicted_labels = []
    actual = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = best_model.predict(data)
            actual.extend(target.cpu().numpy())
            predicted_labels.extend(pred)
            correct += (pred == target.cpu().numpy()).sum().item()
            total += target.size(0)

    #Calculate and print results (gets redirected to results.txt)
    print(f"Model: {model_name}\n")
    print(f"Best parameters: {best_params}\n")
    print(f"test accuracy {100*correct/total:.2f}%\n")
    Accuracy = accuracy_score(actual, predicted_labels)
    Precision = precision_score(actual, predicted_labels,average= "macro")
    Sensitivity_recall = recall_score(actual, predicted_labels, average= "macro")
    F1_score = f1_score(actual, predicted_labels, average= "macro")
    Conf_mat = confusion_matrix(actual, predicted_labels)
    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"F1_score":F1_score})
    print("Confusion Matrix: \n")
    print(Conf_mat)
    print("Mean test accuracy from Cross validation \n")
    print(grid_search.cv_results_["mean_test_score"])
    print("Standard deviation of the test accuracy from Cross validation \n")
    print(grid_search.cv_results_["std_test_score"])

if __name__ == "__main__":
    # Choose the model on the command line:
    #   python main.py            -> CNN
    #   python main.py logreg     -> logistic regression
    parser = argparse.ArgumentParser(description="Cross validation on AMDNet23")
    parser.add_argument("model", nargs="?", default="cnn", choices=["cnn", "logreg"])
    args = parser.parse_args()

    #Creates a results text file from  the print statements in the functions
    results_path = Path(__file__).resolve().parent.parent / f"results/Project3_results_{args.model}.txt"
    with results_path.open("w", encoding="utf-8") as results_file:
        with redirect_stdout(results_file):
            CNN_test(args.model)
