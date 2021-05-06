import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from statistics import mean
from torch import nn, optim
from torchvision.utils import save_image

from models import Generator, Discriminator
from dataset import VtubersImagesDataset

# 利用するエンジンを決定
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# GeneratorとDiscriminatorのインスタンスを生成
model_G = Generator().to(device)
model_D = Discriminator().to(device)

# それぞれのモデルのパラメータを設定
params_G = optim.Adam(model_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
params_D = optim.Adam(model_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# 潜在特徴100次元ベクトルz
noise_z = 100
batch_size = 64

# ロスを計算するときのラベル変数
ones = torch.ones(batch_size).to(device) # 正 1
zeros = torch.zeros(batch_size).to(device) # 負 0
loss_f = nn.BCEWithLogitsLoss()

# 潜在特徴変数z(適当な乱数100次元)
check_z = torch.randn(batch_size, noise_z, 1, 1).to(device)

# dataset読み込み
# # バッチサイズ
batch_size = 64
dataset = VtubersImagesDataset("./img/training_images-64/", transforms.Compose([transforms.ToTensor()]))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(device)
for epoch in range(300):
    log_loss_G = []
    log_loss_D = []
    
    for real_img in tqdm.tqdm(data_loader):
        batch_len = len(real_img)

        # == Generatorの訓練 ==
        # 偽画像を生成
        z = torch.randn(batch_len, noise_z, 1, 1).to(device)
        fake_img = model_G(z)

        # 偽画像の値をキャスト
        fake_img_tensor = fake_img.detach()

        # 実画像として損失を計算
        out = model_D(fake_img)
        loss_G = loss_f(out, ones[: batch_len])
        log_loss_G.append(loss_G.item())

        model_D.zero_grad()
        model_G.zero_grad()
        loss_G.backward()
        params_G.step()

        # == Discriminatorの訓練 ==
        real_img = real_img.to(device)

        # 実画像として損失を計算
        real_out = model_D(real_img)
        loss_D_real = loss_f(real_out, ones[: batch_len])

        # 計算結果をキャスト
        fake_img = fake_img_tensor

        # 偽画像として損失を計算
        fake_out = model_D(fake_img_tensor)
        loss_D_fake = loss_f(fake_out, zeros[: batch_len])

        # 実画像と偽画像のロスの合計
        loss_D = loss_D_real + loss_D_fake
        log_loss_D.append(loss_D.item())

        model_D.zero_grad()
        model_G.zero_grad()
        loss_D.backward()
        params_D.step()

        mean(log_loss_G), mean(log_loss_D)

    if epoch % 10 == 0:
        torch.save(model_G.state_dict(), './weight/Weight_Gen{:03d}.prm'.format(epoch), pickle_protocol=4)
        torch.save(model_D.state_dict(), './weight/Weight_Dis{:03d}.prm'.format(epoch), pickle_protocol=4)

        generated_img = model_G(check_z) 
        save_image(generated_img, './img/generated_images-64/{:03d}.jpg'.format(epoch))