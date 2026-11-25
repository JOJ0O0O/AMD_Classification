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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
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

# ImageNet channel statistics, used as default normalisation for RGB images.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def print_epoch_log(text):
    """
    Print so the text of the progress does not get put into results .txt
    """
    print(text, file=sys.stderr)

class FitProgress(Callback):
    """Update the shared progress bar after each model fit.

    Date: 2026-11-22.
    Written by: AI assistant using Copilot SDK in VS Code.
    Add this reusable progress-bar callback to the AMD classification project for the cross validation.
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
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, list[str]]:
    """
    Load the combined AMDNet23 images and create a stratified 80/20 split.

    Date: 2026-11-19.
    Written by: AI assistant using Copilot SDK in VS Code.
    Add this reusable image-loading function to the AMD
    classification project.
    This corresponds to Level 3 usage of AI, as I tested and validated this function myself

    Date: 2026-11-25.
    Modified by: Claude Opus 5.5 (Anthropic). Aspect-ratio preserving resize
    with centre crop, per-channel normalisation, corrected (height, width)
    order in the docstring.
    Level: TODO - set after testing and validating the changes yourself.

    Date: 2026-11-25.
    Modified by: AI assistant using Copilot SDK in VS Code. Load from the
    combined image folder and create deterministic stratified training and
    validation subsets in an 80/20 ratio.
    Level: TODO - set after testing and validating these changes yourself.

    Images of any size and colour mode are accepted: ImageFolder converts
    every image to RGB, the shorter side is scaled so that the image covers
    image_size without distortion, and the centre is cropped to image_size.

    Args:
        folder: The images directory containing ``combined``, or the combined
            image directory itself. Relative paths are resolved from the
            project root.
        batch_size: Number of images in each data-loader batch.
        image_size: Output (height, width) of every image.
        num_workers: Number of worker processes used to load data.
        mean: Per-channel mean used for normalisation.
        std: Per-channel standard deviation used for normalisation.

    Returns:
        A tuple containing the training loader, validation loader, and class
        names in dataset order.

    Raises:
        ValueError: If the folder or combined dataset is missing, if any class
            has fewer than five images, or if the dataset contains no images.
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

    combined_folder = (
        images_folder
        if images_folder.name.lower() == "combined"
        else images_folder / "combined"
    )
    if not combined_folder.is_dir():
        raise ValueError(f"Combined images folder does not exist: {combined_folder}")

    # Resize(int) scales the SHORTER side and keeps the aspect ratio. Using the
    # larger target dimension guarantees that both sides are at least as large
    # as image_size, so CenterCrop never has to pad. Normalize then gives every
    # channel roughly zero mean and unit variance.
    transform = transforms.Compose(
        [
            transforms.Resize(max(image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # ImageFolder infers class labels from subdirectory names and opens every
    # image as RGB, so grayscale/RGBA files also become 3-channel tensors.
    dataset = datasets.ImageFolder(combined_folder, transform=transform)
    if not dataset.samples:
        raise ValueError(f"No images found in {combined_folder}")

    targets = np.asarray(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(dataset.classes))
    if np.any(class_counts < 5):
        raise ValueError("Each class must contain at least five images to split")

    # The first fold of a shuffled five-fold stratified split gives each class
    # an 80/20 train/validation split while keeping the result reproducible.
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)
    train_indices, valid_indices = next(
        splitter.split(np.zeros(len(targets)), targets)
    )

    class StratifiedImageSubset(torch.utils.data.Subset):
        def __init__(
            self, image_folder: datasets.ImageFolder, indices: np.ndarray
        ) -> None:
            super().__init__(image_folder, indices.tolist())
            self.classes = image_folder.classes
            self.targets = [
                image_folder.targets[index] for index in self.indices
            ]

    train_dataset = StratifiedImageSubset(dataset, train_indices)
    valid_dataset = StratifiedImageSubset(dataset, valid_indices)

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

def run_cross_validation(model_name="cnn"):
    """
    Function to do a Cross validation of the selected model ("cnn" or "logreg")
    for the AMDNet23 dataset.
    The parameters which are validated are set in cross_val_config.py

    Date: 2026-11-25.
    Modified by: Claude Opus 5.5 (Anthropic). Generalised from CNN_test() so
    the same cross validation runs for the CNN and the logistic regression.
    Level: TODO - set after testing and validating the changes yourself.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Generate image loader
    train_loader, test_loader, class_names = load_images(folder="images", batch_size=64, image_size=image_size, num_workers=0)

    #Select model, its constructor arguments and the parameter grid
    module, module_kwargs, model_param_grid = build_module(model_name, train_loader, class_names)

    #Set up Scorch NeuralNetClassifier with the selected model
    Classifier = NeuralNetClassifier(
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
        Classifier,
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
    labels = list(range(len(class_names)))
    Precision = precision_score(actual, predicted_labels, average="macro", zero_division=0)
    Sensitivity_recall = recall_score(actual, predicted_labels, average="macro", zero_division=0)
    F1_score = f1_score(actual, predicted_labels, average="macro", zero_division=0)
    Conf_mat = confusion_matrix(actual, predicted_labels, labels=labels)
    print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"F1_score":F1_score})
    print("Per-class and averaged classification report (recall is sensitivity):")
    print(
        classification_report(
            actual,
            predicted_labels,
            labels=labels,
            target_names=class_names,
            zero_division='warn',
        )
    )
    print("Confusion Matrix (rows = actual, columns = predicted):")
    print("Actual \\ Predicted\t" + "\t".join(class_names))
    for class_name, row in zip(class_names, Conf_mat):
        print(f"{class_name}\t" + "\t".join(str(count) for count in row))
    print("Mean test accuracy from Cross validation:")
    print(grid_search.cv_results_["mean_test_score"])
    print("Standard deviation of the test accuracy from Cross validation:")
    print(grid_search.cv_results_["std_test_score"])

if __name__ == "__main__":
    # Choose the model on the command line:
    #   python main.py            -> CNN
    #   python main.py logreg     -> logistic regression
    parser = argparse.ArgumentParser(description="Cross validation on AMDNet23")
    parser.add_argument("model", nargs="?", default="cnn", choices=["cnn", "logreg"])
    args = parser.parse_args()

    #Creates a results text file from  the print statements in the functions
    results_path = Path(__file__).resolve().parent.parent / f"results/Project3_results_{args.model}_wo_distorted_images.txt"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as results_file:
        with redirect_stdout(results_file):
            run_cross_validation(args.model)