# ===============================================================
# High-Resolution (128×128) Version of Variance-Aware Image Pipeline
# Sharper Originals + Reconstructions
# ===============================================================



import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------------- Preprocessing ----------------------
def gaussian_kernel(kernel_size=11, sigma=2.0, channels=1):
    ax = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1., device=device)
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
    kernel = kernel / torch.sum(kernel)
    return kernel[None,None,:,:].repeat(channels,1,1,1)

def low_pass(x, k): return F.conv2d(x, k, groups=x.shape[1], padding=k.shape[-1]//2)
def high_pass(x, k): return x - low_pass(x, k)

def sobel_edges(x):
    if x.shape[1]==3:
        gray = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    else: gray=x
    gx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], device=device).float().view(1,1,3,3)
    gy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], device=device).float().view(1,1,3,3)
    Gx, Gy = F.conv2d(gray,gx,padding=1), F.conv2d(gray,gy,padding=1)
    mag = torch.sqrt(Gx**2 + Gy**2 + 1e-8)
    mmin, mmax = mag.amin((1,2,3),True), mag.amax((1,2,3),True)
    return (mag-mmin)/(mmax-mmin+1e-8)

def simple_lbp(x):
    if x.shape[1]==3:
        gray = 0.2989*x[:,0:1]+0.5870*x[:,1:2]+0.1140*x[:,2:3]
    else: gray=x
    pad = F.pad(gray,(1,1,1,1),mode="replicate")
    offs=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    lbp = torch.zeros_like(gray)
    for i,(dy,dx) in enumerate(offs):
        nb=pad[:,:,1+dy:1+dy+gray.size(2),1+dx:1+dx+gray.size(3)]
        lbp+=(nb>=gray).float()*(1<<i)
    return lbp/255.0

def preprocess(imgs):
    B,C,H,W = imgs.shape
    gk = gaussian_kernel(11,3.0,C)
    hp = high_pass(imgs,gk)
    lbp = simple_lbp(imgs)
    texture = torch.cat([hp, lbp], dim=1)
    light = torch.mean(low_pass(imgs,gk), dim=1, keepdim=True)
    edges = sobel_edges(imgs)
    boundary = torch.clamp(edges + torch.randn_like(edges)*0.03,0,1)
    return texture, light, boundary

# ---------------------- Encoders ----------------------
class TextureEncoderConvNeXt(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        self.align = nn.Conv2d(4,3,kernel_size=1)
        base = models.convnext_tiny(weights="IMAGENET1K_V1").features
        self.backbone = nn.Sequential(*list(base.children())[:7])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(768, emb_dim)
    def forward(self,x):
        x = self.align(x)
        with torch.no_grad():
            f = self.backbone(x)
        return self.fc(self.pool(f).view(x.size(0),-1))

class LightEncoderVGG(nn.Module):
    def __init__(self, emb_dim=64):
        super().__init__()
        self.align = nn.Conv2d(1,3,kernel_size=1)
        base = models.vgg16_bn(weights="IMAGENET1K_V1").features
        self.backbone = nn.Sequential(*list(base.children())[:10])
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, emb_dim)
    def forward(self,x):
        x = self.align(x)
        with torch.no_grad():
            f = self.backbone(x)
        return self.fc(self.pool(f).view(x.size(0),-1))

class BoundaryEncoderResNet(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.align = nn.Conv2d(1,3,kernel_size=1)
        base = models.resnet18(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool, base.layer1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64, emb_dim)
    def forward(self,x):
        x = self.align(x)
        with torch.no_grad():
            f = self.backbone(x)
        return self.fc(self.pool(f).view(x.size(0),-1))

# ---------------------- Decoder (Improved for 128x128) ----------------------
class UpscaleDecoder(nn.Module):
    def __init__(self, emb_dim, out_ch=3):
        super().__init__()
        self.fc = nn.Linear(emb_dim,128*8*8)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),
            nn.ConvTranspose2d(16,8,4,2,1),nn.ReLU(),
            nn.Conv2d(8,out_ch,3,1,1),nn.Sigmoid()
        )
    def forward(self,z):
        x = self.fc(z).view(z.size(0),128,8,8)
        return self.net(x)

# ---------------------- Full Model ----------------------
class VarianceAwareModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_tex = TextureEncoderConvNeXt()
        self.enc_light = LightEncoderVGG()
        self.enc_bound = BoundaryEncoderResNet()
        self.proj_l_to_b = nn.Linear(64,32)
        self.alpha_net = nn.Sequential(nn.Linear(96,64), nn.ReLU(), nn.Linear(64,1), nn.Sigmoid())
        self.tag_net = nn.Sequential(nn.Linear(224,128), nn.ReLU(), nn.Linear(128,3), nn.Softmax(dim=1))
        self.decoder = UpscaleDecoder(224)
    def forward(self,imgs):
        tex,light,bound = preprocess(imgs)
        zt = self.enc_tex(tex)
        zl = self.enc_light(light)
        zb = self.enc_bound(bound)
        zl_to_b = self.proj_l_to_b(zl)
        alpha = self.alpha_net(torch.cat([zl,zb],dim=1))
        zb_mix = alpha*zl_to_b + (1-alpha)*zb
        fused = torch.cat([zt,zl,zb_mix],dim=1)
        gates = self.tag_net(fused)
        g_t,g_l,g_b = gates[:,0:1],gates[:,1:2],gates[:,2:3]
        zt_g, zl_g, zb_g = zt*g_t, zl*g_l, zb_mix*g_b
        final = torch.cat([zt_g,zl_g,zb_g],dim=1)
        out = self.decoder(final)
        return out,(alpha,gates)

# ---------------------- Training ----------------------
def train_model(epochs_warm=2, epochs_fine=3, batch_size=64, lr=1e-3, entropy_beta=0.02):
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor()
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data",train=True,download=True,transform=transform)
    loader = torch.utils.data.DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=2,drop_last=True)
    model = VarianceAwareModel().to(device)
    opt = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fn = nn.MSELoss()

    for p in model.enc_tex.parameters(): p.requires_grad=False
    for p in model.enc_light.parameters(): p.requires_grad=False
    for p in model.enc_bound.parameters(): p.requires_grad=False
    print("Stage 1 – Warm-up")
    for ep in range(epochs_warm):
        total=0
        for imgs,_ in tqdm(loader,leave=False):
            imgs=imgs.to(device)
            recon,(alpha,gates)=model(imgs)
            loss=loss_fn(recon,imgs)
            p=torch.clamp(gates,1e-9,1.0)
            ent=-torch.sum(p*torch.log(p),dim=1).mean()
            total_loss=loss - entropy_beta*ent
            opt.zero_grad(); total_loss.backward(); opt.step()
            total+=total_loss.item()
        print(f"Epoch {ep+1}: {total/len(loader):.5f}")

    for p in model.parameters(): p.requires_grad=True
    opt=torch.optim.Adam(model.parameters(),lr=1e-5)
    print("Stage 2 – Fine-tune")
    for ep in range(epochs_fine):
        total=0
        for imgs,_ in tqdm(loader,leave=False):
            imgs=imgs.to(device)
            recon,(alpha,gates)=model(imgs)
            loss=loss_fn(recon,imgs)
            p=torch.clamp(gates,1e-9,1.0)
            ent=-torch.sum(p*torch.log(p),dim=1).mean()
            total_loss=loss - entropy_beta*ent
            opt.zero_grad(); total_loss.backward(); opt.step()
            total+=total_loss.item()
        print(f"Fine Epoch {ep+1}: {total/len(loader):.5f}")
    return model

# ---------------------- Visualization ----------------------
def show_images(imgs, titles=None):
    n=len(imgs); fig,axs=plt.subplots(1,n,figsize=(14,3))
    for i,ax in enumerate(axs):
        arr=imgs[i].detach().cpu().numpy()
        if arr.ndim==3 and arr.shape[0]<=4: arr=np.transpose(arr,(1,2,0))
        ax.imshow(np.clip(arr,0,1), interpolation='nearest')
        ax.axis("off")
        if titles: ax.set_title(titles[i])
    plt.show()

# ---------------------- Main Execution ----------------------
if __name__ == '__main__':
    # ---------------------- Train ----------------------
    model = train_model(epochs_warm=2, epochs_fine=3)

    testset = torchvision.datasets.CIFAR10(root="./data",train=False,download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((128,128)),
                                               transforms.ToTensor()
                                           ]))
    imgs,_ = next(iter(torch.utils.data.DataLoader(testset,batch_size=4,shuffle=True)))
    imgs = imgs.to(device)
    with torch.no_grad():
        recon,(alpha,gates)=model(imgs)
    for i in range(4):
        print(f"Sample {i} | alpha {alpha[i].item():.3f} | gates {gates[i].cpu().numpy()}")
        tex,light,bound=preprocess(imgs[i:i+1])
        show_images([imgs[i],
                     tex[0].mean(0,keepdim=True).repeat(3,1,1),
                     light[0].repeat(3,1,1),
                     bound[0].repeat(3,1,1),
                     recon[i]],
                     ["Original","Texture","Light","Boundary","Recon"])
