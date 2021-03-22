from pathlib import Path

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class VtubersImagesDataset(Dataset):
    IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".ping"]

    def __init__(self, img_dir, transform=None):
        # 画像ファイルのパス一覧を取得する。
        self.img_paths = self._get_img_paths(img_dir)
        self.transform = transform

    def __getitem__(self, index):
        path = self.img_paths[index]

        # 画像を読み込む。
        img = Image.open(path)

        if self.transform is not None:
            img = self.transform(img)

        return img

    def _get_img_paths(self, img_dir):
        img_dir = Path(img_dir)
        img_paths = [p for p in img_dir.iterdir() if p.suffix in VtubersImagesDataset.IMG_EXTENSIONS]

        return img_paths

    def __len__(self):
        return len(self.img_paths)



if __name__ == '__main__':# Transform を作成する。
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    # Dataset を作成する。
    dataset = VtubersImagesDataset("./img/training_images/", transform)
    # DataLoader を作成する。
    dataloader = DataLoader(dataset, batch_size=3)

    for batch in dataloader:
        print(batch.shape)