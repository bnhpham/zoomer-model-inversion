from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.utils import save_image


"""
Implementation of the paper "Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion" by Yin, Hongxu, et al. (2020)

Paper:
https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html

Github:
https://github.com/NVlabs/DeepInversion/tree/master

"""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

pytorch_seed = torch.initial_seed()
print(f"PyTorch Initial Seed: {pytorch_seed}")

# Mean & Std of ImageNet
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1,3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1,3,1,1)

# TV loss (Copied from Github Repo (cf. get_image_prior_losses()))
def compute_tv_loss(x):
    diff1 = x[:, :, :, :-1] - x[:, :, :, 1:]        # horizontal neighbours
    diff2 = x[:, :, :-1, :] - x[:, :, 1:, :]        # vertical neighbours
    diff3 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]     # diagonal
    diff4 = x[:, :, :-1, :-1] - x[:, :, 1:, 1:]     # diagonal

    loss_var_l2 = (torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4))
    loss_var_l1 = ((diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()) * 255.0

    return loss_var_l1, loss_var_l2

# Compute competition loss for single-logit binary classifiers 
# (almost same implementaion as the paper's one. However, the paper implements it for multi-class classifcation)
def compute_competition_loss(logit_teacher, logit_student, T=3.0):

    kl_loss = nn.KLDivLoss(reduction='batchmean')

    # Use temperature T to soften probability distributions
    P = torch.sigmoid(logit_student / T)
    Q = torch.sigmoid(logit_teacher / T)

    # Concatenate with complement to make 2-class distribution
    P = torch.cat([P, 1 - P], dim=1).clamp(0.01, 0.99)
    Q = torch.cat([Q, 1 - Q], dim=1).clamp(0.01, 0.99)

    # Midpoint of teacher and student distributions
    M = (0.5 * (P + Q)).clamp(0.01, 0.99)

    # Compute Jensen-Shannon divergence
    js_div = 0.5 * kl_loss(torch.log(P), M) + 0.5 * kl_loss(torch.log(Q), M)

    # Competition loss = 1 - js_div
    return 1.0 - torch.clamp(js_div, 0.0, 1.0)


def random_jitter(x, max_shift=8):
    dy = torch.randint(-max_shift, max_shift+1, (1,), device=x.device).item()
    dx = torch.randint(-max_shift, max_shift+1, (1,), device=x.device).item()
    return torch.roll(torch.roll(x, dy, dims=2), dx, dims=3)


# Feature hook to track statistics
class FeatureHook:

    def __init__(self, module, weight=1.0):
        self.weight = weight
        
        # Difference between current statistics and stored BN statistics
        self.r_feature = torch.tensor(0.0, device=DEVICE)

        # Once the BN layer is finished with processing, call the _fn method,
        # which computes the distance between output statistics and the saved BN statistics
        self.hook = module.register_forward_hook(self._fn)

    def _fn(self, module, inp, out):
        # Output of the previous layer before the BN layer
        x = inp[0]

        # Mean & variance of output
        mean = x.mean(dim=[0,2,3])
        var  = x.var(dim=[0,2,3], unbiased=False)

        # Compute the distance between output statistics and the saved BN statistics
        self.r_feature = self.weight * (torch.norm(mean - module.running_mean, 2) + torch.norm(var  - module.running_var,  2))

    def remove(self):
        self.hook.remove()


# (Adaptive) DeepInversion
def adaptive_deep_inversion(teacher, student, student_opt, target_class, steps, lr, bn_weight, tv_l2_weight, tv_l1_weight, l2_weight, adi_scale,
                            batch_size, jitter, first_bn_multiplier=10.0, multi_res_syn=True):

    PIXEL_MIN = ((torch.zeros(1,3,1,1,device=DEVICE) - MEAN) / STD)     # Normalized representation of black pixels
    PIXEL_MAX = ((torch.ones(1,3,1,1,device=DEVICE)  - MEAN) / STD)     # Normalized representation of white pixels
    
    # FeatureHook
    hooks     = []
    bn_layers = [m for m in teacher.modules() if isinstance(m, nn.BatchNorm2d)]
    for i, m in enumerate(bn_layers):
        # The authors of the paper gave extra weight to the very first BN layer using first_bn_multiplier
        #w = first_bn_multiplier if i == 0 else 1.0

        # We tested whether assigning higher weight to the subsequent BN layers would lead to better results. We observed no significant difference.
        w = first_bn_multiplier if i == 0 or i == 1 or i == 2 or i == 3 else 1.0

        # We also tested what happens if the last BN layers were assigned with greater weights --> no significant difference.
        #w = first_bn_multiplier if i >= len(bn_layers) - 4 else 1.0

        hooks.append(FeatureHook(m, weight=w))

    current_res = 112 if multi_res_syn else 224                                             # Resolution of images
    inputs = torch.randn(batch_size, 3, current_res, current_res, device=DEVICE) * 0.1      # Random initialization
    target = torch.full((batch_size, 1), float(target_class), device=DEVICE)                # Set target class for which we want to generate images
    inputs = inputs.to(DEVICE).requires_grad_(True)                                         # Track gradients w.r.t. the pixel values

    img_opt = torch.optim.Adam([inputs], lr=lr, betas=(0.5, 0.9))                           # Optimizer
    img_sch = torch.optim.lr_scheduler.CosineAnnealingLR(img_opt, T_max=steps)              # Scheduler
    criterion = nn.BCEWithLogitsLoss()                                                      # Classfication loss

    for step in tqdm(range(steps)):
        
        # Upscale resolution of images if multi-resolution synthesis is applied
        if step == steps // 2 and multi_res_syn:
            with torch.no_grad():
                # Upsample in normalized space
                upsampled = F.interpolate(inputs.detach(), size=(224, 224), mode='nearest') # We also tried bilinear interpolation --> no significant improvement
            inputs = upsampled.requires_grad_(True)
            
            # Reinitialization
            img_opt = torch.optim.Adam([inputs], lr=lr, betas=(0.5, 0.9))
            img_sch = torch.optim.lr_scheduler.CosineAnnealingLR(img_opt, T_max=steps // 2)

        # Apply jitter and random flip as augmentation techniques
        img_aug = random_jitter(inputs, jitter // (1 if inputs.shape[-1] == 224 else 2))
        if torch.rand(1).item() > 0.5:
            img_aug = torch.flip(img_aug, dims=(3,))
        
        # If images haven't been scaled up yet, match models input expectation (224x224)
        if img_aug.shape[-1] == 112:
            img_aug = F.interpolate(img_aug, size=(224, 224), mode='nearest')

        # Teacher prediction
        t_logit = teacher(img_aug)
        p_t = torch.sigmoid(t_logit).detach()
        conf = p_t.mean().item() if target_class == 1 else (1 - p_t).mean().item()

        # Classification & BN loss
        cls_loss = criterion(t_logit, target)
        bn_loss = sum(h.r_feature for h in hooks)

        # Image prior
        loss_var_l1, loss_var_l2 = compute_tv_loss(inputs)
        image_prior = tv_l2_weight * loss_var_l2 + tv_l1_weight * loss_var_l1 + l2_weight * torch.norm(inputs.view(batch_size, -1), dim=1).mean()

        # Competition loss (only used in Adaptive DeepInversion)
        compete_loss = torch.tensor(0.0, device=DEVICE)
        if adi_scale > 0.0:
            s_logit_compete = student(img_aug)

            # Jensen-Shannon Divergence
            compete_loss = compute_competition_loss(t_logit.detach(), s_logit_compete, T=3.0)

            # Train student
            s_logit_train = student(img_aug)

            with torch.no_grad():
                soft_target = torch.sigmoid(teacher(img_aug))

            loss_student  = F.binary_cross_entropy_with_logits(s_logit_train, soft_target)
            student_opt.zero_grad()
            loss_student.backward()
            student_opt.step()
        
        total_loss = cls_loss + bn_weight * bn_loss + image_prior + adi_scale * compete_loss

        # Backpropagate loss
        img_opt.zero_grad()
        total_loss.backward()
        img_opt.step()
        img_sch.step()
        with torch.no_grad():
            inputs.data.clamp_(PIXEL_MIN, PIXEL_MAX)

    for h in hooks:
        h.remove()

    final_img = (inputs.detach() * STD + MEAN).clamp(0, 1).clone()
    return final_img, conf


def extract_training_data(teacher, target_class, steps,
                    lr, bn_weight, tv_l2_weight, tv_l1_weight, l2_weight, adi_scale,
                    batch_size, jitter, out_dir, 
                    first_bn_multiplier, multi_res_syn):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Student model learns generated images and serves as a memory
    # Competition loss between teacher and student encourages new samples out of the student's learned knowledge --> broader coverage of train data
    student = resnet18(num_classes=1).to(DEVICE)
    student.train()
    student_opt = torch.optim.Adam(student.parameters(), lr=1e-3)

    # Load Student. If no checkpoint is available, start from scratch
    student_path = Path(out_dir) / "adi_checkpoint.pt"

    if student_path.exists():
        ckpt = torch.load(student_path, map_location=DEVICE)
        start_idx = ckpt["next_idx"]
        student.load_state_dict(ckpt["student"])
        student_opt.load_state_dict(ckpt["student_opt"])
    else:
        start_idx = 0

    print(f"student memory: {start_idx} samples seen so far")

    # DeepInversion
    img, conf = adaptive_deep_inversion(
        teacher=teacher, student=student,
        student_opt=student_opt,
        target_class=target_class,
        steps=steps, lr=lr,
        bn_weight=bn_weight, tv_l2_weight=tv_l2_weight, tv_l1_weight=tv_l1_weight,
        l2_weight=l2_weight, adi_scale=adi_scale,
        batch_size=batch_size, jitter=jitter,
        first_bn_multiplier=first_bn_multiplier,
        multi_res_syn=multi_res_syn,
    )

    print(f"Save {img.shape[0]} images...")
    current_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    for b_idx in range(img.shape[0]):
        individual_path = (out_dir / f"{current_time}_adi_{start_idx:02d}_sample_{b_idx}_target{target_class}_conf{conf:.3f}.png")
        save_image(img[b_idx].clamp(0, 1), individual_path)
        
    # Save student
    next_idx = start_idx + batch_size if adi_scale > 0.0 else start_idx
    torch.save({"next_idx" : next_idx,
                "student": student.state_dict(),
                "student_opt" : student_opt.state_dict(),
                }, student_path)


def main():

    out_dir = "adi_results"
    target_class = 0
    steps = 4000                # 4000, 7000, 10000 or 20000 were typically used during the challenge
    lr = 0.05                   # either 0.01 or 0.05
    bn_weight = 0.05
    tv_l2_weight = 1e-4
    tv_l1_weight = 0.0
    l2_weight = 1e-5
    adi_scale = 0.2             # either 0.0, 0.2, 2.0 or 10.0 (set to 0.0 for normal DeepInversion)
    batch_size = 64            # either 8, 16, 32, 64, 128 or 160 (bigger sizes not possible due to limited VRAM)
    jitter = 8                  # either 8, 16 or 32
    first_bn_mult = 10.0        # either 0.0 or 10.0
    multi_res_syn = True

    # Load teacher/target model
    teacher = resnet18(num_classes=1)
    state = torch.load("extracted_model.pth", map_location="cpu")
    teacher.load_state_dict(state, strict=False)

    # Freeze target model
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    extract_training_data(
        teacher=teacher,
        target_class=target_class,
        steps=steps,
        lr=lr,
        bn_weight=bn_weight,
        tv_l2_weight=tv_l2_weight,
        tv_l1_weight=tv_l1_weight,
        l2_weight=l2_weight,
        adi_scale=adi_scale,
        batch_size=batch_size,
        jitter=jitter,
        out_dir=out_dir,
        first_bn_multiplier=first_bn_mult,
        multi_res_syn=multi_res_syn,
    )


if __name__ == "__main__":
    main()