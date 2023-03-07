import math
from operator import mod
from sqlalchemy import null
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
from torchvision.transforms import Resize
import torch.nn.functional as F

from models.gaussian_diffusion import get_named_beta_schedule
from models.respace import SpacedDiffusion, space_timesteps


def resize_tensor(input_tensor):
    width=input_tensor.shape[2]
    height = input_tensor.shape[3]
    output_tensor = input_tensor
    output_tensor_list = []
    output_tensor_list.append(output_tensor)
    for i in range(3):
        width = width//2
        height = height//2
        tensor = output_tensor
        torch_resize_fun = Resize([width,height])
        output_tensor = torch_resize_fun(tensor)
        output_tensor_list.insert(0, output_tensor)

    return output_tensor_list

NUM_CLASSES = 1000

from .guided_diffusion_modules.unet_improved import UNetModel

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    # how unet constructed
    if image_size == 256: 
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif image_size == 64:
        channel_mult = (1, 2, 3, 4)
    elif image_size == 32:
        channel_mult = (1, 2, 2, 2)
    else:
        raise ValueError(f"unsupported image size: {image_size}")

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        in_channels=6,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6), #如果学习sigma的话，输出是6个通道，前三个是通道预测eps噪声，后三个通道预测方差
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )

class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        super(Network, self).__init__(**kwargs)
        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet_modified import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net

        elif module_name == 'transformer':
            from .transformer_modules.timeswinir import TimeSwinIR
            self.denoise_fn = TimeSwinIR(**unet) #去噪模型是一个u-net

        elif module_name == 'wavelet':
            from .guided_diffusion_modules.unet_wavelet_skip import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net
        elif module_name == 'focal':
            from .guided_diffusion_modules.unet_modified_focal_attn import UNet
            self.denoise_fn = UNet(**unet) #去噪模型是一个u-net
        elif module_name == 'noise_level_estimation':
            from .guided_diffusion_modules.unet_modified_with_est import UNet
            self.denoise_fn = UNet(**unet)
        elif module_name == 'improved':
            from .guided_diffusion_modules.unet import UNet
            self.denoise_fn = UNet(**unet)
        
        elif module_name == 'improved_biggan':
            model_defaults = dict(
                image_size=256,
                num_channels=128,
                num_res_blocks=3,
                learn_sigma=True,
                class_cond=False,
                use_checkpoint=False,
                attention_resolutions="16,8",
                num_heads=4,
                num_heads_upsample=-1,
                use_scale_shift_norm=True,
                dropout=0.0,)
            self.denoise_fn = create_model(**model_defaults)
        
        self.beta_schedule = beta_schedule
        self.num_timesteps = beta_schedule['train']['n_timestep']

        #print(self.num_timesteps)
        if not beta_schedule['test']['is_test']:
            self.time_step_respacing = self.num_timesteps
            self.spaced_dpm = self._create_gaussian_diffusion(steps=self.num_timesteps, noise_schedule='squaredcos_cap_v2')
        else:
            self.time_step_respacing = beta_schedule['test']['time_step_respacing']
            self.spaced_dpm = self._create_gaussian_diffusion(steps=self.num_timesteps, noise_schedule='squaredcos_cap_v2', timestep_respacing=str(self.time_step_respacing))

    def _create_gaussian_diffusion(self, steps, noise_schedule, timestep_respacing=''):
        betas = get_named_beta_schedule(noise_schedule, steps)
        if not timestep_respacing:
            timestep_respacing = [steps]
        return SpacedDiffusion(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
        )

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        """
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        #gammas_prev = np.append(1., gammas[:-1])


        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        """
        pass

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn


    def q_sample(self, y_0, time_step, noise=None): #计算扩散过程中任意时刻y_t的采样值，直接套公式,采样得到一张噪声图片
        return self.spaced_dpm.q_sample(x_start=y_0, t=time_step, noise=noise)



    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None, model_kwargs=None): #从y_t采样t时刻的重构值
        # Pack the tokens together into model kwargs. 用字典来保存模型参数，提高了模型接口的可扩展性

        #需要在这里把相关的参数给整理好
        def keep_background_unchange_fn(x_t):
            # Force the model to have the exact right x_start predictions
            # for the part of the image which is known.
            return (
                x_t * model_kwargs['foreground_mask']
                + model_kwargs['original_image'] * (1. - model_kwargs['foreground_mask'])
            )

        out = self.spaced_dpm.p_sample(model=self.denoise_fn, x=y_t, t=t, clip_denoised=clip_denoised, denoised_fn=keep_background_unchange_fn, cond_fn=None,model_kwargs=model_kwargs)
        
        image = out["sample"]

        return image

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8): #采样过程，类似于IDDPM中的源码p_sample_loop
        b, *_ = y_cond.shape

        assert self.time_step_respacing > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.time_step_respacing//sample_num)
        
        model_kwargs = dict(

            #mask=mask,#torch.Size([2, 128])

            # Masked inpainting image
            y_cond=y_cond,
            foreground_mask=mask,
            original_image=y_0
        )
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.time_step_respacing)), desc='sampling loop time step', total=self.time_step_respacing):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond, model_kwargs=model_kwargs) #将y_t作为下一个迭代的输入来生成新的y_t #会在p_sample调用函数。
            #if mask is not None:
                #y_t = y_0*(1.-mask) + mask*y_t #得到y_t之后，将y_t作为下一个sample 生成的输入
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None): #参数顺序，真实图片y_0，合成图片(条件图片y_cond)以及mask
        # sampling from p(gammas)该函数的输出是loss，可以调用IDDPM中的compute_losses函数来实现。
        
        ###构造t
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long() #随机生成一个时间点

        #构造可变参数
        model_kwargs = dict(

            mask= mask,#torch.Size([2, 128])

            # Masked inpainting image
            y_cond=y_cond,
            #inpaint_mask=source_mask_64.repeat(full_batch_size, 1, 1, 1).to(device),
            noise=noise,
        )
        
        #torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0]
        
        loss = self.spaced_dpm.training_losses(self.denoise_fn, y_0, t, model_kwargs=model_kwargs)
        
        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


if __name__ == "__main__":
    dpm = create_gaussian_diffusion(steps=1000, noise_schedule='linear', timestep_respacing="100")
    print(dpm)