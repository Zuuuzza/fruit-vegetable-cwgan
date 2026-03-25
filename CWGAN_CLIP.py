#WEIGHT CLIPPING
import os
import random
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
SAVE_IMAGE_INTERVAL = 15  # interwał zapisywania obrazów
LR = 1e-4  # learning rate
BATCH_SIZE = 64  # rozmiar batcha
CRITIC_ITERATIONS = 5  # ile razy uczony jest krytyk (dyskryminator) na jedną aktualizację generatora
EMBED_DIM = 100  # wymiar wektora etykiety
FEATURES_GEN = 64  # liczba „detektorów cech” w generatorze
FEATURES_CRITIC = 64  # liczba „detektorów cech” w krytyku

SAMPLES_FOLDER = "Owoce_wyniki/obrazki_cwganGP_weight_clip"
MODELS_FOLDER = "Owoce_wyniki/model_cwganGP_weight_clip"
DATA_PATH = "./owoce"

os.makedirs(SAMPLES_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacja obrazów do rozmiaru 64x64, konwersja na Tensor i normalizacja do zakresu [-1, 1]
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Wczytanie danych z folderu
dataset = ImageFolder(root=DATA_PATH, transform=transform)

# Generator
class cGenerator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g, embed_dim, num_classes):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        # Sieć generująca obrazek z szumu i zakodowanej etykiety
        self.net = nn.Sequential(
            self._block(z_dim + embed_dim, features_g * 8, 4, 1, 0),  # 1x1 -> 4x4
            self._block(features_g * 8, features_g * 4, 4, 2, 1),     # 4x4 -> 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1),     # 8x8 -> 16x16
            self._block(features_g * 2, features_g, 4, 2, 1),         # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g, channels_img, 4, 2, 1),    # 32x32 -> 64x64
            nn.Tanh(),  # Zakres pikseli [-1, 1]
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, noise, labels):
        # Łączenie szumu i zakodowanej etykiety
        label_emb = self.label_embedding(labels)
        x = torch.cat([noise, label_emb], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(x)

# Krytyk
class cCritic(nn.Module):
    def __init__(self, channels_img, features_d, embed_dim, num_classes, img_size=64):
        super().__init__()
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.img_size = img_size
        self.net = nn.Sequential(
            nn.Conv2d(channels_img + embed_dim, features_d, 4, 2, 1),  # 64x64 -> 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 32x32 -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 16x16 -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 8x8 -> 4x4
            nn.Conv2d(features_d * 8, 1, 4, 1, 0),  # 4x4 -> 1x1 (pojedynczy wynik)
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.LeakyReLU(0.2),
    )


    def forward(self, x, labels):
        # Łączenie obrazka z zakodowaną etykietą
        N = x.size(0)
        label_emb = self.label_embedding(labels).unsqueeze(2).unsqueeze(3)
        label_emb = label_emb.expand(N, -1, self.img_size, self.img_size)
        x = torch.cat([x, label_emb], dim=1)
        return self.net(x)

# zapisywanie obrazków
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

# trening z klipowaniem wag
def train_cwgan(gen, critic, loader, epochs, lr, critic_iters, clip_value=0.01):
    # Optymalizatory 
    opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)

    # przeniesienie modeli na GPU
    gen.to(device)
    critic.to(device)

    # pętla treningowa po epokach
    for epoch in range(1, epochs + 1):
        g_loss_epoch = 0
        c_loss_epoch = 0

        # iteracja po batchach
        for batch_idx, (real, labels) in enumerate(loader):
            real = real.to(device)
            labels = labels.to(device)
            cur_batch_size = real.size(0)

            # tym razem aktualizujemy krytyka kilka razy przed aktualizacją generatora
            for _ in range(critic_iters):
                z = torch.randn(cur_batch_size, LATENT_DIM, device=device)
                fake = gen(z, labels)

                # Oceny dla prawdziwych i fałszywych obrazków
                crit_real = critic(real, labels).reshape(-1)
                crit_fake = critic(fake.detach(), labels).reshape(-1)

                #  WGAN loss dla krytyka
                loss_critic = -(crit_real.mean() - crit_fake.mean())
                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                #  Klipowanie wag krytyka
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # aktualizacja generatora
            z = torch.randn(cur_batch_size, LATENT_DIM, device=device)
            fake = gen(z, labels)
            gen_fake = critic(fake, labels).reshape(-1)

            # WGAN loss dla generatora
            loss_gen = -gen_fake.mean()
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # Sumowanie strat dla epoki
            c_loss_epoch += loss_critic.item()
            g_loss_epoch += loss_gen.item()

        print(f"==> EPOCH {epoch}/{epochs} | Critic Loss: {c_loss_epoch/len(loader):.4f}, Generator Loss: {g_loss_epoch/len(loader):.4f}")

        # Zapisywanie obrazków co kilka epok
        if epoch % SAVE_IMAGE_INTERVAL == 0:
            for label in range(NUM_CLASSES):
                save_sample_images(gen, epoch, label)

def main():
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    gen = cGenerator(LATENT_DIM, CHANNELS_IMG, FEATURES_GEN, EMBED_DIM, NUM_CLASSES)
    crit = cCritic(CHANNELS_IMG, FEATURES_CRITIC, EMBED_DIM, NUM_CLASSES, img_size=IMAGE_SIZE)

    print("\nTraining final model...")
    train_cwgan(gen, crit, dataloader, epochs=EPOCHS, lr=LR, critic_iters=CRITIC_ITERATIONS)

if __name__ == "__main__":
    main()
