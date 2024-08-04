# AI_OF_GOD_V2
This is our submission for the ai_of_god_v2 competition. Team members:- Alok Raj,Anant Upadhyay,Manav Jain
## Image Classification with EfficientNet

This project demonstrates an image classification pipeline using the EfficientNet model with the PyTorch framework. The dataset consists of images that are processed and augmented before being fed into the model for training and validation.

## Requirements

- Python 3.10
- PyTorch
- Timm
- Albumentations
- OpenCV
- PIL
- Pandas
- NumPy
- Matplotlib
- TorchMetrics
- Scikit-learn

Install the required packages using:
```
pip install timm torch albumentations opencv-python pillow pandas numpy matplotlib torchmetrics scikit-learn
```
## Dataset
The dataset is expected to be in the following structure:
```
/kaggle/input/ai-of-god-v20/
    train/
        0.jpg
        1.jpg
        ...
    train.csv
```
- FileName: Name of the image file
- Class: Class label of the image
### Data Augmentation and Transformation
The dataset undergoes several transformations including resizing, cropping, normalization, and flipping. Albumentations library is used for these transformations.
```
transform = A.Compose(
    [
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ]
)
```
### Custom Dataset Class
A custom dataset class is defined to handle the loading and transformation of images.
```
class CustomDataset(Dataset):
    def __init__(self, folder, labels):
        self.transform = transform
        self.folder = folder
        self.plabels = pd.read_csv(labels)
        self.lis = np.sort(os.listdir(folder))
        self.labels = np.array(pd.read_csv(labels))

    def __len__(self):
        return self.plabels.shape[0]

    def __getitem__(self, idx):
        y = torch.tensor((np.array(self.plabels[self.plabels['FileName'] == self.lis[idx]]['Class'])), dtype=torch.long)
        img = cv.cvtColor(np.array(Image.open(self.folder + '/' + self.lis[idx])), cv.COLOR_GRAY2RGB)
        x = self.transform(image=img)
        x['image'] = (x['image'].type(torch.float32)).to(device)
        return {'x': x['image'], 'y': y.to(device)}

    def labels_counts(self):
        a, b = np.unique(self.labels[:, 1], return_counts=True)
        size = {}
        for i in range(a.size):
            size[a[i]] = b[i]
        return size
```
### Data splitting
The dataset is split into training and validation sets.
```
data = CustomDataset('/kaggle/input/ai-of-god-v20/train', '/kaggle/input/ai-of-god-v20/train.csv')
train, val = random_split(data, [7000, len(data) - 7000])
```
### Model
An EfficientNet model from the timm library is used.
```
model_timm = timm.create_model("efficientnet_b0", pretrained=True)
```
<img src="https://wisdomml.in/wp-content/uploads/2023/03/eff_banner-768x358.png">

### Training and Evaluation
The model is trained and evaluated using PyTorch's training loop.

#### Usage
- Clone the repository and navigate to the project directory.
- Ensure the dataset is structured as described above.
- Install the required packages.
- Run the notebook or script.


