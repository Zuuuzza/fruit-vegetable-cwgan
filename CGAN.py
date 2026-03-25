# CGAN
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

EPOCHS = 150  # liczba epok treningowych
LATENT_DIM = 100  # rozmiar wejścia generatora
IMAGE_SIZE = 64  # rozmiar obrazka wejściowego/wyjściowego (64x64 piksele)
CHANNELS_IMG = 3  # liczba kanałów (3 = RGB)
NUM_CLASSES = 3  # liczba klas 
SAVE_IMAGE_INTERVAL = 15  
LR = 1e-4  # learning rate
BATCH_SIZE = 64  # rozmiar batcha
EMBED_DIM = 100  # wymiar wektora etykiety
FEATURES_GEN = 64  # liczba  cech w warstwach konwolucyjnych w generatorze
FEATURES_DISC = 64  # liczba  cech w warstwach konwolucyjnych w dyskryminatorze

SAMPLES_FOLDER = "Owoce_wyniki/obrazki_cgan"
MODELS_FOLDER = "Owoce_wyniki/model_cgan"
DATA_PATH = "./owoce"

os.makedirs(SAMPLES_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Tranformacja obrazów do formatu 64 x 64, stworzenie Tensora z 3 kanałami (RGB) i normalizacja wartości pikseli do zakresu [-1, 1]
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

#pobranie danych i przetransformowanie ich do formatu Tensora zdefiniowanego przez transform
dataset = ImageFolder(root=DATA_PATH, transform=transform)

#Generator
class cGenerator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, embed_dim, num_classes):
        super().__init__()
        # zmiana etykiet na wektory o stałej długości
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        # sieć generująca obrazek z szumu + etykiety
        self.net = nn.Sequential(
            # Warstwa 1:
            # Wejście: [batch_size, 200, 1, 1] (szum + etykieta)
            # Wyjście: [batch_size, 512, 4, 4]
            # Z obrazka 1x1 pikseli (szum + etykieta) robi się obrazek 4x4 pikseli z 512 kanałami
            self._block(z_dim + embed_dim, features_g * 8, 4, 1, 0),
            # Warstwa 2:
            # Wejście: [batch_size, 512, 4, 4]
            # Wyjście: [batch_size, 256, 8, 8]
            # Obrazek rośnie do  8x8 pikseli, z 256 kanałami
            self._block(features_g * 8, features_g * 4, 4, 2, 1),
            # Warstwa 3:
            # Wejście: [batch_size, 256, 8, 8]
            # Wyjście: [batch_size, 128, 16, 16]
            # Obrazek rośnie do 16x16 pikseli, z 128 kanałami
            self._block(features_g * 4, features_g * 2, 4, 2, 1),
            # Warstwa 4:
            # Wejście: [batch_size, 128, 16, 16]
            # Wyjście: [batch_size, 64, 32, 32]
            #  Obrazek rośnie do 32x32 pikseli, z 64 kanałami
            self._block(features_g * 2, features_g, 4, 2, 1),
            # Ostatnia warstwa (bez BatchNorm i ReLU):
            # Wejście: [batch_size, 64, 32, 32]
            # Wyjście: [batch_size, 3, 64, 64]
            #  Przekształca obrazek do formatu 64x64 pikseli z 3 kanałami
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1),
            # Przekształca piksele do zakresu [-1, 1], żeby pasowały do normalizacji obrazów
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            # dekonwolucja - przekształca obrazek z mniejszej rozdzielczości do większej
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            # normalizacja
            nn.BatchNorm2d(out_c),
            # aktywacja ReLU
            nn.ReLU(True),
        )

    def forward(self, noise, labels):
        # Łączenie szumu i etykiet w jeden tensor
        label_emb = self.label_embedding(labels)
        x = torch.cat([noise, label_emb], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(x)

# Dyskryminator
class cDiscriminator(nn.Module):
    def __init__(self, channels_img, features_d, embed_dim, num_classes, img_size=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.img_size = img_size
        # sieć oceniająca (czy obrazek jest prawdziwy i zgodny z etykietą)
        self.net = nn.Sequential(
            # Warstwa 1:
            # Wejście: [batch_size, 3 + 100, 64, 64] (obrazek + etykieta)
            # Wyjście: [batch_size, 64, 32, 32]
            # Z obrazka 64x64 pikseli (3 kanały + etykieta) robi się obrazek 32x32 pikseli z 64 kanałami
            nn.Conv2d(channels_img + embed_dim, features_d, 4, 2, 1),
            # Aktywacja LeakyReLU - to ReLU, ale z małym nachyleniem dla wartości ujemnych
            nn.LeakyReLU(0.2),
            # Warstwa 2:
            # Wejście: [batch_size, 64, 32, 32]
            # Wyjście: [batch_size, 128, 16, 16]
            # Obrazek rośnie do 16x16 pikseli, z 128 kanałami
            self._block(features_d, features_d * 2, 4, 2, 1),
            # Warstwa 3:
            # Wejście: [batch_size, 128, 16, 16]
            # Wyjście: [batch_size, 256, 8, 8]
            # Obrazek rośnie do 8x8 pikseli, z 256 kanałami
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # Warstwa 4:
            # Wejście: [batch_size, 256, 8, 8]
            # Wyjście: [batch_size, 512, 4, 4]
            # Obrazek rośnie do 4x4 pikseli, z 512 kanałami
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # Ostatnia warstwa (bez BatchNorm i ReLU):
            # Wejście: [batch_size, 512, 4, 4]
            # Wyjście: [batch_size, 1, 1, 1]
            #  Przekształca obrazek do pojedynczej wartości (prawdziwy/fałszywy)
            nn.Conv2d(features_d * 8, 1, 4, 1, 0),
            # Aktywacja Sigmoid - przekształca wynik do zakresu [0, 1]
            nn.Sigmoid(),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            # konwolucja - przekształca obrazek z większej rozdzielczości do mniejszej
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            # normalizacja
            nn.BatchNorm2d(out_c),
            # aktywacja LeakyReLU
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, labels):
        # Łączenie obrazka i etykiet w jeden tensor
        N = x.size(0)
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(N, -1, self.img_size, self.img_size)
        x = torch.cat([x, label_emb], dim=1)
        return self.net(x)

#zapisywanie obrazków
def save_sample_images(gen, epoch, label, folder=SAMPLES_FOLDER, n=8):
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(n, LATENT_DIM, device=device)
        labels = torch.full((n,), label, dtype=torch.long, device=device)
        fake = gen(noise, labels).cpu()
    gen.train()
    label_name = dataset.classes[label]
    fig, axs = plt.subplots(1, n, figsize=(n * 2.5, 2.5))
    for i, ax in enumerate(axs):
        img = fake[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2.0
        ax.imshow(img.clip(0, 1))
        ax.set_title(label_name)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"epoch_{epoch}_label_{label_name}.png"))
    plt.close()

# trenowanie cGAN
def train_cgan(gen, disc, loader, epochs, lr):
    # Binary Cross Entropy Loss
    criterion = nn.BCELoss()
    # Optymalizatory dla generatora i dyskryminatora
    opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    # przeniesienie się na kartę graficzną
    gen.to(device)
    disc.to(device)
    criterion.to(device)

    # pętla treningowa po epokach
    for epoch in range(1, epochs + 1):
        g_loss_epoch = 0
        d_loss_epoch = 0

        # iteracja po batchach
        for real, labels in loader:
            # przeniesienie danych na kartę graficzną
            real = real.to(device)
            labels = labels.to(device)
            cur_batch_size = real.size(0)

            # etykiety dla prawdziwych i fałszywych obrazów
            real_labels = torch.ones(cur_batch_size, 1, device=device)
            fake_labels = torch.zeros(cur_batch_size, 1, device=device)

            # trening dyskryminatora
            # generowanie szumu i fałszywych obrazów
            noise = torch.randn(cur_batch_size, LATENT_DIM, device=device)
            fake = gen(noise, labels)

            # przewidywanie prawdziwych i fałszywych obrazów
            real_pred = disc(real, labels).view(-1, 1)
            fake_pred = disc(fake.detach(), labels).view(-1, 1)

            # obliczanie strat dyskryminatora
            loss_real = criterion(real_pred, real_labels)
            loss_fake = criterion(fake_pred, fake_labels)
            loss_disc = (loss_real + loss_fake) / 2

            # gradienty i aktualizacja dyskryminatora
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # trening generatora
            fake_pred = disc(fake, labels).view(-1, 1)
            loss_gen = criterion(fake_pred, real_labels)

            # gradienty i aktualizacja generatora
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # sumowanie strat dla epoki
            d_loss_epoch += loss_disc.item()
            g_loss_epoch += loss_gen.item()

        print(f"==> EPOCH {epoch}/{epochs} | D Loss: {d_loss_epoch/len(loader):.4f}, G Loss: {g_loss_epoch/len(loader):.4f}")

        if epoch % SAVE_IMAGE_INTERVAL == 0:
            for label in range(NUM_CLASSES):
                save_sample_images(gen, epoch, label)

def main():
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gen = cGenerator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN, EMBED_DIM, NUM_CLASSES)
    disc = cDiscriminator(CHANNELS_IMG, FEATURES_DISC, EMBED_DIM, NUM_CLASSES, img_size=IMAGE_SIZE)

    print("\nTraining cGAN...")
    train_cgan(gen, disc, dataloader, epochs=EPOCHS, lr=LR)

if __name__ == "__main__":
    main()
