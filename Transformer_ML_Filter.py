#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:25:46 2024

@author: astronomy_zrf

该程序用于处理数据与模型读取，为Project的核心程序

需要的参量: 归一化参数，模型本身

"""



import math
import numpy as np
import torch
import warnings

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from moudel.mgmcformer import MgMcFORMER
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from collections import Counter
from scipy.interpolate import interp1d
from scipy.interpolate import PchipInterpolator
from data.preprocessing import toDataloader_1
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gpytorch

warnings.filterwarnings("ignore")


def plot_interpolation_debug(
    g_time_sorted, g_flux_sorted,
    r_time_sorted, r_flux_sorted,
    new_time_g, g_interp,
    new_time_r, r_interp,
    idx=None, filename=None, save=False,
    save_dir="debug_plots", invert_y=False,
    label=0
):
    plt.figure(figsize=(8, 4))
    plt.plot(g_time_sorted, g_flux_sorted, '.', label='g-band (raw)', alpha=0.6)
    plt.plot(r_time_sorted, r_flux_sorted, '.', label='r-band (raw)', alpha=0.6)
    plt.plot(new_time_g, g_interp, '-', label='g-band (interp)', linewidth=1.2)
    plt.plot(new_time_r, r_interp, '-', label='r-band (interp)', linewidth=1.2)

    plt.xlabel("Time")
    plt.ylabel("Flux")
    plt.legend()

    title_str = f"Sample #{idx}, Label={label}" if idx is not None else ""
    if filename is not None:
        title_str += f" | {filename}"
    plt.title(title_str.strip())

    if invert_y:
        plt.gca().invert_yaxis()

    plt.tight_layout()

    if save:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fname = filename.replace(".json", "") if filename else f"sample_{idx}"
        plt.savefig(f"{save_dir}/{fname}.png")
    else:
        plt.show()

   

def mean_standardize_fit(X):
    m1 = np.mean(X, axis=1)
    mean = np.mean(m1, axis=0)

    s1 = np.std(X, axis=1)
    std = np.mean(s1, axis=0)

    return mean, std


def mean_standardize_transform(X, mean, std):
    return (X - mean) / std


def preprocess(X_train,mean,std):
    mean, std = mean,std
    X_train = mean_standardize_transform(X_train, mean, std)

    X_train_task = torch.as_tensor(X_train).float()
    

    return X_train_task



def load_model(model_name = "test_model_ZTF.pth",model_type='normal'):
    import argparse
    import utils
    #先试用完全加载的模型
    
    
    if model_type == 'normal':
        #baseline
        mean_1 = np.array([2.58541981 ,0.03422268, 3.48849084, 0.04991302])
        std_1 = np.array([ 8.4326514 ,  0.13886326 ,10.67072193 , 0.16889779])
        
        #ref1
        #mean_1 = np.array([1.36207055, 0.02676976, 1.25395946 ,0.03292777])
        #std_1 = np.array([[4.40499441, 0.12267711 ,4.36520069, 0.14686071]])
        
        #ref2
        # mean_1 = np.array([0.54075231, 0.01707357, 0.64735728, 0.02409625])
        # std_1 = np.array([2.4671549 , 0.09101748 ,2.8297367 , 0.11766426])
        
        #mut-stage-res
        #mean_1 = np.array([1.49290933, 0.03445279, 2.0073296 , 0.0567213 ])
        #std_1 = np.array([5.62762415, 0.15126424, 6.99387466, 0.20028241])
    
    elif model_type == 'interp':
        mean_1 = [22.92619544,  0.60272906, 24.38933124,  0.70646293]
        std_1 = [11.66679564,  0.17765913, 13.25543993,  0.16187295]
    
    elif model_type == 'interp_GP':
        #mean_1 = [23.75334653,  0.5954749 , 23.75334653,  0.70871375]
        #std_1 = [13.78397299,  0.1632627 , 13.78397299,  0.14765083]
        mean_1 = [25.3968,  0.5987, 26.5292,  0.6900]
        std_1 = [13.1211,  0.1882, 14.3865,  0.1758]
        
    else:
        mean_1 = np.array([0.23500219, 0.02121095, 0.24013096 ,0.02216163] )
        std_1 = np.array([1.50556694 ,0.12050369 ,1.52651072, 0.12612386])
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ZTF_target(纯粹原始数据)')
    parser.add_argument('--multi_group', type=list, default=[1],
                        help='Input list')  # group<=math.ceil(sqrt(seq_len))
    parser.add_argument('--batch', type=int, default=1, help='Dataset batch')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--emb_size', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--emb_size_c', type=int, default=32)
    parser.add_argument('--masking_ratio', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--ratio_highest_attention', type=float, default=0.35)
    #parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.01)
    parser.add_argument('--nhid', type=int, default=32)
    parser.add_argument('--nhid_c', type=int, default=32)
    args = parser.parse_args(args=[])
    prop = utils.get_prop(args)
    prop['multi_group'] = [int(patch_index) for patch_index in prop['multi_group']]
    prop['batch_true'] = 1
    prop['nclasses'] = 6
    prop['seq_len'], prop['input_size'] = 200, 4
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
   
    
    
    
    model = MgMcFORMER(prop['multi_group'], prop['nclasses'], prop['seq_len'], prop['input_size'], prop['emb_size'], \
                       prop['nhid'], prop['emb_size_c'], prop['nhid_c'], prop['nhead'], prop['nlayers'], prop['device'],
                       prop['dropout']).to(prop['device'])
    
    
    model.load_state_dict(torch.load(model_name,map_location=torch.device('cpu')))
    #model.load_state_dict(torch.load("data/ZTF_target(早期光变训练集，限制测光点数目大于3)/test_model_ZTF.pth"))
    model.eval()  # Set to evaluation mode
    
    return model,mean_1,std_1

def test_model(model, dataloader_test, nclasses, device):
    model.eval()  # Turn on the evaluation mode

    output_arr = []
    label_arr = []
    with torch.no_grad():
        for data, label in dataloader_test:
            data = data.to(device)
            
            label = label.to(device)
            #print(data[0])
            #print(model(data)[0])
            pred = model(data)[0]
            #print(model(data))
            
            output_arr.append(pred)
            label_arr.append(label)
            

    return label_arr,output_arr


def load_data(lc_g,lc_r):
    tg,fg = lc_g[0],lc_g[1]
    tr,fr = lc_r[0],lc_r[1]
    
    target_tg = np.zeros(200,dtype='float32');target_fg = np.zeros(200,dtype='float32')
    target_tr = np.zeros(200,dtype='float32');target_fr = np.zeros(200,dtype='float32')
    
    max_f = max(np.max(fg),np.max(fr))
    min_t = min(np.min(tg),np.min(tr))
    
    for i in range(min(len(tg),200)):
        target_tg[i] = tg[i]-min_t
        target_fg[i] = fg[i]/max_f
    for i in range(min(len(tr),200)):
        target_tr[i] = tr[i]-min_t
        target_fr[i] = fr[i]/max_f
    
    final_data = np.vstack((target_tg , target_fg ,target_tr , target_fr )).T
    #print(final_data[0])
    
    final_data=np.expand_dims(final_data, axis=0)
    
    return final_data


class MLPInterpolator(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.model(x)
        

class LSTMInterpolator(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len=1, input_size=1)
        out, _ = self.lstm(x)
        return self.fc(out)[:, -1, :]  # output shape: (batch, 1)


def interpolate_with_mlp(time_array, flux_array, new_time, epochs=2000, lr=1e-3, hidden_size=64, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 去除 NaN
    mask = ~np.isnan(flux_array)
    x = time_array[mask].reshape(-1, 1)
    y = flux_array[mask].reshape(-1, 1)

    if len(x) < 3:
        return PchipInterpolator(time_array, flux_array)(new_time)

    # 归一化
    #t_min, t_max = x.min(), x.max()
    #x_norm = (x - t_min) / (t_max - t_min)
    #new_time_norm = (new_time.reshape(-1, 1) - t_min) / (t_max - t_min)

    # 新 Gaussian 归一化
    t_mean, t_std = x.mean(), max(x.std(), 1e-3)
    x_norm = (x - t_mean) / t_std
    new_time_norm = (new_time.reshape(-1, 1) - t_mean) / t_std

    y_mean, y_std = y.mean(), max(y.std(), 1e-3)
    y_norm = (y - y_mean) / y_std

    x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=device)

    model = MLPInterpolator(hidden_size=hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(x_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()

    # 预测
    new_time_tensor = torch.tensor(new_time_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_norm = model(new_time_tensor).cpu().numpy()

    pred = pred_norm * y_std + y_mean
    return np.clip(pred.flatten(), 1e-6, np.max(flux_array) * 10)


def safe_multiband_gp_interpolate(
    g_time, g_flux, r_time, r_flux,
    new_time_g, new_time_r,
    epochs=1000,
    device="cuda",
    fallback_threshold=0.95,
    patience=200,
    min_delta=1e-3,
    loss_threshold_for_accept=99,  # <<< 新增：GP 收敛质量标准
    verbose=True
):
    import numpy as np
    import torch
    import gpytorch
    #from interpolate_with_mlp import interpolate_with_mlp  # 确保你实现了这个函数

    # 预测退化判断函数
    def prediction_is_almost_constant(pred_values, tolerance=1e-2):
        return np.std(pred_values) < tolerance


    class MultibandGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=2)
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # 1. 数据拼接
    g_band = np.zeros_like(g_time)
    r_band = np.ones_like(r_time)

    all_time = np.concatenate([g_time, r_time])
    all_flux = np.concatenate([g_flux, r_flux])
    all_band = np.concatenate([g_band, r_band])

    train_x_np = np.stack([all_time, all_band], axis=-1)
    time_min, time_max = train_x_np[:, 0].min(), train_x_np[:, 0].max()
    train_x_np[:, 0] = (train_x_np[:, 0] - time_min) / (time_max - time_min + 1e-8)

    train_x = torch.tensor(train_x_np, dtype=torch.float32).to(device)
    train_y = torch.tensor(all_flux, dtype=torch.float32).to(device)

    # 2. 模型初始化
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = MultibandGPModel(train_x, train_y, likelihood).to(device)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    best_loss = float('inf')
    patience_counter = 0
    losses = []
    converged = False

    # 3. 训练过程 + Early Stopping
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        if verbose and epoch % 100 == 0:
            print(f"[Multiband GP][Epoch {epoch}] Loss: {current_loss:.4f}")

        if best_loss - current_loss > min_delta:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"[Multiband GP] 提前早停于第 {epoch} 轮，Best Loss: {best_loss:.4f}")
            converged = True
            break

    # 若未早停，再根据下降程度判断是否算“收敛”
    if not converged:
        #if losses[-1] < fallback_threshold * losses[0] and losses[-1] <= 0:
        if losses[-1] < fallback_threshold * losses[0]:
            best_loss = losses[-1]
            converged = True
        else:
            converged = False

    # 4. 输入标准化后的时间坐标
    new_x_g = np.stack([new_time_g, np.zeros_like(new_time_g)], axis=-1)
    new_x_r = np.stack([new_time_r, np.ones_like(new_time_r)], axis=-1)

    new_x_g[:, 0] = (new_x_g[:, 0] - time_min) / (time_max - time_min + 1e-8)
    new_x_r[:, 0] = (new_x_r[:, 0] - time_min) / (time_max - time_min + 1e-8)

    # 5. 判断是否使用 GP 插值或 fallback 到 MLP
    if converged and best_loss < loss_threshold_for_accept:
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            new_x_g_tensor = torch.tensor(new_x_g, dtype=torch.float32).to(device)
            new_x_r_tensor = torch.tensor(new_x_r, dtype=torch.float32).to(device)

            pred_g = likelihood(model(new_x_g_tensor)).mean.cpu().numpy()
            pred_r = likelihood(model(new_x_r_tensor)).mean.cpu().numpy()

            # ==== 新增判断：预测是否退化为常数 ====
        if (prediction_is_almost_constant(pred_g) and
            prediction_is_almost_constant(pred_r)):
            if verbose:
                print(f"[Multiband GP] 预测退化为均值，Fallback 到 MLP。")
            pred_g = interpolate_with_mlp(np.array(g_time), np.array(g_flux), new_time_g, device=device)
            pred_r = interpolate_with_mlp(np.array(r_time), np.array(r_flux), new_time_r, device=device)
            return pred_g, pred_r

        return pred_g, pred_r

    else:
        if verbose:
            print(f"[Multiband GP] Fallback 到 MLP。收敛状态: {converged}, 最终 Loss: {best_loss:.4f}")
        pred_g = interpolate_with_mlp(np.array(g_time), np.array(g_flux), new_time_g, device=device)
        pred_r = interpolate_with_mlp(np.array(r_time), np.array(r_flux), new_time_r, device=device)
        return pred_g, pred_r


def load_data_interp(lc_g,lc_r,interp_type='normal'):
    tg,fg = lc_g[0],lc_g[1]
    tr,fr = lc_r[0],lc_r[1]
    
    # 对 g 和 r 波段的时间和流量进行排序
    sorted_g_indices = np.argsort(tg)  # 获取 g_time 排序的索引
    g_time_sorted = tg[sorted_g_indices]  # 根据排序的索引排序 g_time
    g_flux_sorted = fg[sorted_g_indices]  # 同时排序 g_flux
    
    sorted_r_indices = np.argsort(tr)  # 获取 r_time 排序的索引
    r_time_sorted = tr[sorted_r_indices]  # 根据排序的索引排序 r_time
    r_flux_sorted = fr[sorted_r_indices]  # 同时排序 r_flux
    
   
    
    # 删除 g 和 r 波段中时间相同的点
    unique_g_times = []
    unique_g_flux = []
    unique_r_times = []
    unique_r_flux = []
    
    for g_time_val, g_flux_val in zip(g_time_sorted, g_flux_sorted):
        if g_time_val not in unique_g_times:
            unique_g_times.append(g_time_val)
            unique_g_flux.append(g_flux_val)
    
    for r_time_val, r_flux_val in zip(r_time_sorted, r_flux_sorted):
        if r_time_val not in unique_r_times:
            unique_r_times.append(r_time_val)
            unique_r_flux.append(r_flux_val)
    
    # 重新组合已去除重复时间点的 g 和 r 波段
    g_time_sorted = np.array(unique_g_times)
    g_flux_sorted = np.array(unique_g_flux)
    r_time_sorted = np.array(unique_r_times)
    r_flux_sorted = np.array(unique_r_flux)
    
    tg = g_time_sorted;tr = r_time_sorted
    fg = g_flux_sorted;fr = r_flux_sorted
    
    max_min_time = max(np.min(tg),np.min(tr))
    min_min_time = min(np.min(tg),np.min(tr))
    min_max_time = min(np.max(tg),np.max(tr))
    
    max_f = max(np.max(fg),np.max(fr))
    
    if interp_type == 'normal':
        new_time_g = np.linspace(np.min(tg),np.max(tg),200)
        new_time_r = np.linspace(np.min(tr),np.max(tr),200)
        
        new_time_full_g = np.linspace(np.min(tg),np.max(tg) , 200)
        new_time_full_r = np.linspace(np.min(tr),np.max(tr) , 200)
    
        
        upper_limit_flux = max(np.max(fg),np.max(fr))
        lower_limit_flux = 1e-6
        
        new_time_g = np.linspace(min(tg),max(tg),200)
        new_time_r = np.linspace(min(tr),max(tr),200)
        
        if len(tg)>=2 and len(tr)>=2:
            pchip_g = PchipInterpolator(tg, fg,extrapolate=True)
            pchip_r = PchipInterpolator(tr, fr,extrapolate=True)
            
            g_interp = pchip_g(new_time_g)
            g_interp = np.clip(g_interp,lower_limit_flux, upper_limit_flux)
            
            r_interp = pchip_r(new_time_r)
            r_interp = np.clip(r_interp,lower_limit_flux, upper_limit_flux)
            
           
            
        elif len(tg)<2 and len(tr)>=3:
            g_interp = np.interp(new_time_g,tg,fg)
            
            pchip_r = PchipInterpolator(tr, fr,extrapolate=True)
            r_interp = pchip_r(new_time_r)
            r_interp = np.clip(r_interp,lower_limit_flux, upper_limit_flux)
            
            
            
        elif len(tr)<2 and len(tg)>=3:
            r_interp = np.interp(new_time_r,tr,fr)
            
            pchip_g = PchipInterpolator(tg, fg,extrapolate=True)
            g_interp = pchip_g(new_time_g)
            g_interp = np.clip(g_interp,lower_limit_flux, upper_limit_flux)
            
        elif len(tr)<2 and len(tg)<2:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
    
        
        
        
    
        
        
        max_f = max(max(g_interp),max(r_interp))
        max_f_full = max(max(g_interp),max(r_interp))
    
        #import matplotlib.pyplot as plt
        #plt.plot(new_time_g,g_interp)
    
        g_interp_full = g_interp
        r_interp_full = r_interp
        
        final_data = np.vstack((new_time_full_g-max_min_time, g_interp_full/max_f_full, new_time_full_r-max_min_time,r_interp_full/max_f_full)).T
        
        final_data=np.expand_dims(final_data, axis=0)
        
        return final_data
    elif interp_type == 'GP':
        
        # new_time_g = np.linspace(0,min_max_time-max_min_time,200)
        # new_time_r = np.linspace(0,min_max_time-max_min_time,200)
        
        new_time_g = np.linspace(np.min(tg),np.max(tg),200)
        new_time_r = np.linspace(np.min(tr),np.max(tr),200)
        
        new_time_g -= min_min_time
        new_time_r -= min_min_time
        
        new_time_full_g = new_time_g
        new_time_full_r = new_time_r
        
        
        
        
        fg=fg/max_f;fr = fr/max_f
        
        tg = tg-min_min_time
        tr = tr-min_min_time
        
        if len(tg)>=4 and len(tr)>=4:
            try:
                g_interp, r_interp = safe_multiband_gp_interpolate(
                            tg, fg,
                            tr, fr,
                            new_time_g, new_time_r,
                            epochs=1000,
                            device="cpu",
                            loss_threshold_for_accept=-0.0,  # <<< 新增：GP 收敛质量标准
                            verbose=False
                        )
            except RuntimeError as e:
                print('GP Process Facing Uncorrectable Error, MLP Instead...')
                g_interp = interpolate_with_mlp(np.array(tg), np.array(fg), new_time_g,device="cpu",epochs=1000)
                r_interp = interpolate_with_mlp(np.array(tr), np.array(fr), new_time_r,device="cpu",epochs=1000)
                
            # global fig
            # global fig_index
            # x=4;y=4;
            # ax = fig.add_subplot(x,y,fig_index)
            # ax.set_title('Candidate')
            # ax.plot(tg,fg,'o',label='real g',color = 'g')
            # ax.plot(tr,fr,'o',label='real r',color = 'r')
            # ax.plot(new_time_g,g_interp,'--',label='fitted g',color = 'g')
            # ax.plot(new_time_r,r_interp,'--',label='fitted r',color = 'r')
            # fig_index+=1
            # fig.subplots_adjust(hspace=0.5)

        elif len(tg)<4 and len(tr)>=4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
            
            
        elif len(tr)<4 and len(tg)>=4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
            
            
        elif len(tr)<4 and len(tg)<4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
            
            
        
        
    elif interp_type == 'MLP':
        
        new_time_g = np.linspace(0,min_max_time-max_min_time,200)
        new_time_r = np.linspace(0,min_max_time-max_min_time,200)
        
        new_time_full_g = np.linspace(max_min_time,min_max_time , 200)
        new_time_full_r = np.linspace(max_min_time,min_max_time , 200)
        
        
        
        fg=fg/max_f;fr = fr/max_f
        
        tg = tg-max_min_time
        tr = tr-max_min_time
        
        if len(tg)>=4 and len(tr)>=4:
           
            g_interp = interpolate_with_mlp(np.array(tg), np.array(fg), new_time_g,device="mps",epochs=1000)
            r_interp = interpolate_with_mlp(np.array(tr), np.array(fr), new_time_r,device="mps",epochs=1000)
            

        elif len(tg)<4 and len(tr)>=4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
            
            
        elif len(tr)<4 and len(tg)>=4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
            
            
        elif len(tr)<4 and len(tg)<4:
            r_interp = np.interp(new_time_r,tr,fr)
            g_interp = np.interp(new_time_g,tg,fg)
        
    
        
        max_f = max(max(g_interp),max(r_interp))
        max_f_full = max(max(g_interp),max(r_interp))
    
        #import matplotlib.pyplot as plt
        #plt.plot(new_time_g,g_interp)
    
    g_interp_full = g_interp
    r_interp_full = r_interp
    
    #print('yes')
    '''
    plot_interpolation_debug(
                                tg, fg,
                                tr, fr,
                                new_time_full_g-np.min(new_time_full_g), g_interp_full,
                                new_time_full_r-np.min(new_time_full_g), r_interp_full,
                                idx=0,
                                save=False,
                                filename=None
                            )
    
    
    '''
    final_data = np.vstack((new_time_full_g, g_interp_full, new_time_full_r,r_interp_full)).T
    
    final_data=np.expand_dims(final_data, axis=0)
    
    return final_data


def model_eval(final_data,model,mean,std):
    final_data = preprocess(final_data,mean,std)
    #print(final_data)
    dataloader_train, dataloader_test = toDataloader_1(final_data,[0], final_data, [0])
    final_data = DataLoader(final_data,batch_size = 1)
    model.eval()
    
    label,output = test_model(model, dataloader_test, nclasses=6, device="cpu")
    probabilities = F.softmax(output[0],dim=1)
    
    return probabilities

def ML_Cks(lc_g,lc_r,model,mean,std,model_type = 'normal'):
    #model,mean,std =  Transformer_ML_Filter.load_model()
    if model_type == 'normal':
        final_data = load_data(lc_g, lc_r)
    elif model_type == 'interp':
        final_data = load_data_interp(lc_g, lc_r)
    elif model_type == 'interp_GP':
        final_data = load_data_interp(lc_g, lc_r,interp_type='GP')
    probabilities = model_eval(final_data, model, mean, std)
    
    return probabilities


def Get_Classification(lc_g,lc_r,model_name = 'test_model_ZTF-Copy3(迁移学习).pth',model_type = 'interp_GP'):
    model,mean,std =  load_model(model_name,model_type=model_type)
    Probs = ML_Cks(lc_g,lc_r,model,mean,std,model_type = model_type)
    return Probs


# 定义类别标签


def classify_tensor(tensor, CLASS_NAMES=["TDE", "SN Ia", "SN Ib/c", "SN II", "SLSN", "AGN"], verbose=False):
    """
    输入：tensor of shape (1, num_classes)
    输出：分类结果字符串
    参数：
        verbose=True: 输出所有类别和对应分数
        verbose=False: 只输出预测类别
    """
    # 如果是 GPU tensor，先转为 CPU
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()

    scores = tensor.squeeze().tolist()
    max_index = torch.argmax(tensor).item()
    predicted_class = CLASS_NAMES[max_index]

    extra_line = ""
    if 0.03 <= scores[0] < 0.5:
        extra_line = "Weak Candidate"
    elif 0.5 <= scores[0] < 0.8:
        extra_line = "Normal Candidate"
    elif scores[0] >= 0.8:
        extra_line = "Strong Candidate"

    if verbose:
        result_lines = [f"{name}: {score:.4f}" for name, score in zip(CLASS_NAMES, scores)]
        return f"Predicted: {predicted_class}\n" + "\n".join(result_lines) + (f"\n{extra_line}" if extra_line else "")
    else:
        return f"{predicted_class}" + (f" ({extra_line})" if extra_line else "")













