#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  9 22:39:57 2025

@author: astronomy_zrf
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 20:42:50 2025

@author: astronomy_zrf
"""

import numpy as np
from scipy.optimize import curve_fit
from functools import partial
import matplotlib.pyplot as plt

import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)
warnings.filterwarnings("ignore")


#定义星等流量转换
def trans_fe2m2(flux,e_flux):
    #AB星等系统，所有的波段都是3631Jy作为零点
    #这个函数用来计算星等误差
    #考虑到读出的流量单位mJy
    if flux>0 and flux - e_flux>0:
        err = -2.5*np.log10((flux-e_flux)/3631000) + 2.5*np.log10((flux+e_flux)/3631000)
    else:
        err = 'inf'
    return err

def trans_f2m(flux):
    if flux>0:
        mag = -2.5*np.log10((flux)/3631000)
        return mag
    else:
        print('nagtive flux')
        return 999



# 定义高斯函数
def gaussian(x, mu, sigma,amp):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))


def gaussian_with_C(x, mu, sigma,amp,C):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))+C

#定义指数拟合
def exp(x,tau,t0,amp):
    return amp * np.exp((x - t0)/tau)

def exp_with_C(x,tau,t0,amp,C):
    return amp * np.exp((x - t0)/tau)+C

#定义SoftPlus函数
def softplus(x, amp,b,t0):
    return amp*(x-t0 + np.sqrt((x-t0)**2 + b))

def gauss_and_exp(x, sigma,tau,t0,amp,C):
    import numpy as np
    return np.piecewise(x, [x<=t0,x>t0], [lambda x:gaussian(x, t0,sigma,amp)+C
                                          ,lambda x:exp(x, tau, t0,amp)+C])



def powerlaw(x,alpha,t0,tp,amp):
    
    return amp*((x-tp+t0)/t0)**(alpha)
    
def gauss_and_powerlaw(x,t0,tp,sigma,alpha,amp,C):
    
    import numpy as np
    return np.piecewise(x, [x<tp,x>=tp], [lambda x:gaussian(x, tp,sigma,amp)+C
                                          ,lambda x:powerlaw(x, alpha, t0,tp,amp)+C])




def gauss_and_powerlaw_fitting(time,flux):
    #根据一片新的文章，这个工作准备对拟合函数进行大调整
    #x,t0,sigma,alpha,amp
    
    alpha = -5/3#-5/3律，设定为初始值
    t0 = 10
    tp = time[np.where(flux == np.max(flux))[0][0]]
    amp = np.max(flux)
    
    sigma = 1
    
    initial_para = [t0,tp,sigma,alpha,amp,0]
    
    bounds = ([1,tp-100,1,-4,0,-0.1],[1000,tp+5,100,-1,2*np.max(flux),0.1])
    
    params,pcov = curve_fit(gauss_and_powerlaw, time, flux, p0 = initial_para,maxfev=8000,bounds=bounds)
    
    residuals = flux - gauss_and_powerlaw(time, *params)
    r_squared = 1 - np.var(residuals) / np.var(flux)
    
    pred_flux = gauss_and_powerlaw(time, *params)
    
    return r_squared,params,pcov,pred_flux

def Gauss_C_fitting(time,flux,t0,f_max,C=0,sigma0=20):

    #mean = np.max(time)
    sigma = sigma0
    amp = abs(f_max)
    c = 0
    tm = t0
    
   
    

    initial_guess = [tm, sigma,amp,c]  # 初始猜测值
    bounds = [[tm-5,1,0.5*amp,-0.1],[np.inf,10**1.5,np.inf,0.1]]
    
    

    params, pcov = curve_fit(gaussian_with_C, time, flux, bounds = bounds,p0=initial_guess,maxfev=6000)
    residuals = flux - gaussian_with_C(time, *params)
    r_squared = 1 - np.var(residuals) / np.var(flux)
    pred_flux = gaussian_with_C(time, *params)

    return r_squared,params,pcov,pred_flux


def exp_fitting(time,flux,t0,f_max,C=0,tau0=-20):
   #def exp(x,tau,t0,amp,C):
   #    return amp * np.exp((x - t0)/tau)+C
    tau = tau0
    t0 = t0
    fm = f_max
    
    initial_exp = [tau,t0,fm,0]
    
    
    
    bounds = [[-1000,-np.inf,0.8*fm,-0.1],[-1,np.inf,np.inf,0.1]]
    
    
    params,pcov = curve_fit(exp_with_C, time, flux,bounds=bounds, p0=initial_exp,maxfev = 1000)
    
    residuals = flux - exp_with_C(time, *params)
    r_squared = 1 - np.var(residuals) / np.var(flux)

    pred_flux = exp_with_C(time, *params)
    
    
    
    return r_squared,params,pcov,pred_flux


def powerlaw_with_c(x,alpha,t0,tp,amp,c):
    
    return amp*((x-tp+t0)/t0)**(alpha)+c

def powerlaw_fitting(time,flux):

    tp = time[np.where(flux == np.max(flux))[0][0]]
    t0 = 10
    amp = np.max(flux)
    alpha = -5/3

    
    initial_para = [alpha,t0,tp,amp,0]
    bounds = [[-4,1,tp-10,0,-0.1],[-1,1000,tp+10,np.inf,0.1]]
    params, pcov = curve_fit(powerlaw_with_c, time, flux, p0=initial_para,maxfev=6000)

    pred_flux = powerlaw_with_c(time,*params)

    residuals = flux - powerlaw_with_c(time, *params)
    r_squared = 1 - np.var(residuals) / np.var(flux)

    return r_squared,params,pcov,pred_flux
    
    
    

def gauss_and_exp_fitting(time,flux,t0,Flag,f_max,C=0,tau0=-20,sigma0=20):

   
    tau = tau0#这个时标指代的是peak之后多少天，该源的流量下降到原先的三分之一
    f_max = abs(f_max)
    #print('t0 = ',t0)
    
    
    sigma = sigma0
    
    initial_para = [sigma,tau,t0,f_max,0]
    
    bounds = ([1,-1000,t0-100,f_max,-0.1],[10**1.5,-1,t0+100,f_max*10,0.1])
   

    #if len(flux)<20:
    #    print('length<10')
    
    
    params,pcov = curve_fit(gauss_and_exp, time, flux,bounds=bounds, p0 = initial_para,maxfev=10000)
    
    residuals = flux - gauss_and_exp(time, *params)
    r_squared = 1 - np.var(residuals) / np.var(flux)

    pred_flux = gauss_and_exp(time, *params)
    
    return r_squared,params,pcov,pred_flux



def cross_point_check_for_fitting_data(lc_g, lc_r, pkt):
    # 判断g, r波段交点是否符合条件，且出现在峰值后一个月内
    
    import numpy as np

    # 提取光变曲线数据
    flux1 = np.array(lc_g[1])  # g波段的flux值
    flux2 = np.array(lc_r[1])  # r波段的flux值
    time = np.array(lc_g[0])   # 共同的时间轴
    
    # 计算两条光变曲线的差值
    diff = flux1 - flux2
    
    # 找到相邻点符号变化的位置（即交点）
    indices = np.where(np.diff(np.sign(diff)) != 0)[0]
    
    # 遍历所有交点位置
    for idx in indices:
        intersection_time = time[idx]
        
        # 检查交点是否出现在峰值后30天内
        if pkt < intersection_time <= pkt + 30:
            # 判断交点之后g波段是否持续小于r波段
            if np.all(flux1[idx + 1:] < flux2[idx + 1:]):
                return False  # 满足条件，返回False
    
    # 如果没有符合条件的交点，返回True
    return True

def photometry_color_forced_test(lc_g, lc_r, pkt,lc_type = 'fitting'):
    # 不基于插值的颜色变化检测，输入的是原始数据
    try:
        from scipy.stats import pearsonr
        import numpy as np

        # 提取g波段和r波段的flux和时间数据
        flux_g = np.array(lc_g[1])
        flux_r = np.array(lc_r[1])
        time_g = np.array(lc_g[0])
        time_r = np.array(lc_r[0])

        # 转换flux为星等
        C = 25
        mag_g = -2.5 * np.log10(flux_g) + C
        mag_r = -2.5 * np.log10(flux_r) + C

        # 设置峰值时间和30天的时间窗口
        peak_time = pkt
        end_time = peak_time + 100

        # 仅选取峰值后30天内的数据
        indices_g = np.where((time_g >= peak_time) & (time_g < end_time))[0]
        indices_r = np.where((time_r >= peak_time) & (time_r < end_time))[0]
        
        post_peak_time_g = time_g[indices_g]
        post_peak_time_r = time_r[indices_r]
        post_peak_mag_g = mag_g[indices_g]
        post_peak_mag_r = mag_r[indices_r]

        # 设置时间窗口
        window_size = 10
        num_segments = int(np.ceil((end_time - peak_time) / window_size))

        # 计算每个时间段的 mag_g - mag_r
        mag_diff_segments = []

        for i in range(num_segments):
            segment_start = peak_time + i * window_size
            segment_end = segment_start + window_size

            # 找到当前时间段内的g和r数据点
            segment_indices_g = np.where((post_peak_time_g >= segment_start) & (post_peak_time_g < segment_end))[0]
            segment_indices_r = np.where((post_peak_time_r >= segment_start) & (post_peak_time_r < segment_end))[0]

            if len(segment_indices_g) > 0 and len(segment_indices_r) > 0:
                # 处理g和r数据点数目不一致的情况
                if len(segment_indices_g) > len(segment_indices_r):
                    mag_diff = []
                    for idx in segment_indices_r:
                        closest_idx = segment_indices_g[np.abs(post_peak_time_g[segment_indices_g] - post_peak_time_r[idx]).argmin()]
                        mag_diff.append(post_peak_mag_g[closest_idx] - post_peak_mag_r[idx])
                    mag_diff_segments.append(np.mean(mag_diff))

                elif len(segment_indices_r) > len(segment_indices_g):
                    mag_diff = []
                    for idx in segment_indices_g:
                        closest_idx = segment_indices_r[np.abs(post_peak_time_r[segment_indices_r] - post_peak_time_g[idx]).argmin()]
                        mag_diff.append(post_peak_mag_g[idx] - post_peak_mag_r[closest_idx])
                    mag_diff_segments.append(np.mean(mag_diff))

                else:
                    # 当g和r数目相同时，直接计算mag_g - mag_r的平均值
                    mag_diff = post_peak_mag_g[segment_indices_g] - post_peak_mag_r[segment_indices_r]
                    mag_diff_segments.append(np.mean(mag_diff))

            else:
                #print('?')
                continue

        # 计算 mag_diff_segments 的变化情况
       
        
        mag_diff_segments = np.array(mag_diff_segments)
        mag_diff_changes = np.diff(mag_diff_segments)
        
        mag_diff_changes = mag_diff_changes[~np.isnan(mag_diff_changes)]

        segment_indices = np.arange(len(mag_diff_changes))
        
        #plt.plot(segment_indices,mag_diff_changes)
        
        #print(mag_diff_segments )
        

        # 相关性分析
        #print(mag_diff_changes)
        if len(segment_indices) >= 2:
            #print(np.mean(mag_diff_changes))
            
            #print(segment_indices,mag_diff_changes)
            
            correlation_coefficient, p_value = pearsonr(segment_indices, mag_diff_changes)
            #print(correlation_coefficient)
        else:
            #print('?')
            return False

        # 返回分析结果
        # print(correlation_coefficient,p_value)
        
        if lc_type=='real':
            #print(correlation_coefficient,p_value)
            return (correlation_coefficient<0.8 and p_value>0.05)#86%
        elif lc_type=='fitting':
            return (p_value>0.05)#86%
    
    except IndexError:
        #print('xxx')
        return False





def cross_point_check_for_real_data(lc_g, lc_r, pkt):
    import numpy as np

    # 提取光变曲线数据
    flux1 = np.array(lc_g[1])  # g波段的flux值
    time1 = np.array(lc_g[0])  # g波段的时间戳
    
    flux2 = np.array(lc_r[1])  # r波段的flux值
    time2 = np.array(lc_r[0])  # r波段的时间戳
    
    # 定义滑动时间窗大小
    window_size = 5  # 单位：天
    intersection_times = []
    
    # 获取所有时间的最小值和最大值
    start_time = pkt
    end_time = min(max(time1), max(time2))
    
    # 从 start_time 开始滑动时间窗，步长为 window_size
    current_time = start_time
    while current_time < end_time:
        # 计算当前时间窗的结束时间
        end_time_window = current_time + window_size
        
        # 提取时间窗内的flux1和flux2数据
        indices1 = np.where((time1 >= current_time) & (time1 < end_time_window))[0]
        indices2 = np.where((time2 >= current_time) & (time2 < end_time_window))[0]
        
        # 检查该时间窗内是否有来自两条曲线的观测数据
        if len(indices1) > 0 and len(indices2) > 0:
            # 获取两条曲线在该窗口的flux值，并对齐时间
            flux1_window = flux1[indices1]
            flux2_window = np.interp(time1[indices1], time2[indices2], flux2[indices2])
            
            # 计算两条曲线的差值
            diff = flux1_window - flux2_window
            
            # 检查是否存在符号变化（交点）
            sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
            for idx in sign_changes:
                intersection_time = time1[indices1][idx]
                
                # 检查交点是否位于峰值后30天以内
                if pkt < intersection_time <= pkt + 40:
                    intersection_times.append(intersection_time)
                    
                   
                    return False  # 满足条件返回False
        
        # 移动到下一个时间窗
        current_time += window_size
    
    # 如果没有符合条件的交点，返回True
    return True


def peak_color_check(lc_g,lc_r):
        '''
        只有在高斯拟合成功的情况下才会进行峰值颜色检测
        '''
        
        pkc = False
        
        tg = lc_g[0]
        fg = lc_g[1]
        
        tr = lc_r[0]
        fr = lc_r[1]
        
        mxf_g = np.max(fg)
        mxf_r = np.max(fr)
        

        
        tpg = tg[np.where(fg == mxf_g)[0][0]]
        tpr = tr[np.where(fr == mxf_r)[0][0]]
        
        if mxf_g - mxf_r>0:
            pkc = True
        
        return pkc

def photometry_color_postpk(lc_g, lc_r, pkt):
    # 仅在峰值后60天内检查最后一个g波段的流量是否大于r波段
    try:
        import numpy as np

        # 提取g波段和r波段的flux和时间数据
        flux_g = np.array(lc_g[1])
        flux_r = np.array(lc_r[1])
        time_g = np.array(lc_g[0])
        time_r = np.array(lc_r[0])

        # 设置峰值时间和60天的时间窗口
        peak_time = pkt
        end_time = peak_time + 100#62%

        # 仅选取峰值后60天内的数据
        indices_g = np.where((time_g >= peak_time) & (time_g < end_time))[0]
        indices_r = np.where((time_r >= peak_time) & (time_r < end_time))[0]
        
        # 获取此时间范围内的g波段和r波段的流量和时间数据
        post_peak_flux_g = flux_g[indices_g]
        post_peak_flux_r = flux_r[indices_r]
        post_peak_time_g = time_g[indices_g]
        post_peak_time_r = time_r[indices_r]

        # 如果时间范围内没有数据，返回False
        if len(post_peak_flux_g) == 0 or len(post_peak_flux_r) == 0:
            return False

        # 获取g波段和r波段最后一个观测点的流量
        last_flux_g = post_peak_flux_g[-1]
        last_flux_r = post_peak_flux_r[-1]
        
       

        # 返回最后一个g波段流量是否大于r波段流量
        
        if last_flux_g<0 or last_flux_r<0:
            return False
        
        return trans_f2m(last_flux_g) - trans_f2m(last_flux_r) <0.05

    except IndexError:
        return False


def compute_aic(rss, n, k):
    """计算 Akaike 信息准则（AIC）"""
    return 2 * k + n * np.log(rss / n)

def compute_mape(y_true, y_pred):
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero]))


def competitive_fitting(time, flux, t0_init, f_max, Flag, sigma0=None, tau0=None):
    if sigma0 is None:
        sigma0 = 10
    if tau0 is None:
        tau0 = -20
    models = {}
    
    

    def safe_fit(name, func):
        try:
            r2, params, pcov, y_pred = func()
            rss = np.sum((flux - y_pred) ** 2)
            k = len(params)
            aic = compute_aic(rss, len(flux), k)
            mape = compute_mape(flux, y_pred)
            models[name] = (aic, params, pcov, rss, r2, mape)
        except RuntimeError:
            models[name] = (np.inf, None, None, None, None, None)

    # 执行拟合
    safe_fit("gauss+exp", lambda: gauss_and_exp_fitting(time, flux, t0_init, Flag=Flag, f_max=f_max, sigma0=sigma0, tau0=tau0))
    safe_fit("gauss", lambda: Gauss_C_fitting(time, flux, t0_init, f_max=f_max))
    safe_fit("exp", lambda: exp_fitting(time, flux, t0_init, f_max=f_max))
    safe_fit("gauss+powerlaw",lambda:gauss_and_powerlaw_fitting(time,flux) )
    safe_fit("powerlaw",lambda:powerlaw_fitting(time,flux))

    # 过滤失败的模型
    valid_models = {k: v for k, v in models.items() if v[1] is not None}

    if not valid_models:
        
        return False, None, None, None, None, None,-np.inf,np.inf  # 全部拟合失败

    # 选择最小 AIC 的模型
    best_name, (best_aic, best_params, best_pcov, best_rss, best_r2, best_mape) = min(valid_models.items(), key=lambda x: x[1][0])

    # 可选打印各模型拟合信息
    # print("\n--- 模型比较 ---")
    # for name, (aic, _, _, _, r2, mape) in models.items():
    #     if r2 is not None:
    #         print(f"{name:10s} | AIC: {aic:.2f} | R²: {r2:.4f} | MAPE: {mape:.2f}%")
    #     else:
    #         print(f"{name:10s} | 拟合失败")

    return True,best_name,best_params, best_pcov, best_rss, best_aic, best_r2, best_mape


def Fitting_criteria_single_band(time, flux):
    t0_init = time[np.argmax(flux)]
    f_max = np.max(flux)
    Flag = 'g'
    Best_Func_Name = None

    success, best_name, best_params, cov, rss, aic,best_r2,best_mape = competitive_fitting(
        time=time,
        flux=flux,
        t0_init=t0_init,
        f_max=f_max,
        Flag=Flag
    )
    if success == False:
        return False
    if success == True:
        if  best_r2>0.6:
            
            if best_name == 'gauss+exp':
                Best_Func_Name = best_name
                sigma,tau,t0,f_max,baseline = best_params

                if sigma>1 and abs(tau)>1:
                    return True,Best_Func_Name,best_params
                else:
                    
                    return False,Best_Func_Name,best_params
            if best_name == 'gauss':
                Best_Func_Name = best_name
                tm, sigma,amp,c = best_params
                if sigma>1:
                    return True,Best_Func_Name,best_params
                else:
                    return False,Best_Func_Name,best_params
            if best_name == 'exp':
                Best_Func_Name = best_name
                tau,t0,fm,c = best_params
                if abs(tau)>1:
                    return True,Best_Func_Name,best_params
                else:
                    
                    return False,Best_Func_Name,best_params
            if best_name == 'gauss+powerlaw':
                Best_Func_Name = best_name
                t0,tp,sigma,alpha,amp,c = best_params
                
                if alpha>-4 and alpha<-1/2 and sigma>1:
                    return True,Best_Func_Name,best_params
                else:
                    return False,Best_Func_Name,best_params
            if best_name == 'powerlaw':
                Best_Func_Name = best_name
                alpha,t0,tp,amp,c = best_params
                if alpha>-4 and alpha<-1/2:
                    return True,Best_Func_Name,best_params
                else:
                    return False,Best_Func_Name,best_params
        else:
            
            return False,Best_Func_Name,best_params


def Parameter_extractor(Best_Func_Name,params):
    #return rising time scale/fading time scale and tpeak
    Rising_scale = None
    Fading_scale = None
    Peak_time = None
    if Best_Func_Name == 'gauss':
        Rising_scale = params[1]

    if Best_Func_Name == 'gauss+exp':
        Rising_scale = params[0]
        Fading_scale = params[1]
        Peak_time = params[2]
    if Best_Func_Name == 'exp':
        Fading_scale = params[0]
        
    if Best_Func_Name == 'gauss+powerlaw':
        Rising_scale = params[2]
        Fading_scale = params[0]
        Peak_time = params[1]
    if Best_Func_Name == 'powerlaw':
        Fading_scale = params[0]
        #powerlaw的情况下峰值时间拥有非常大的不确定性，另外t0似乎对下降的情况并不明显
        #Peak_time = params[2]
    
    return Rising_scale,Fading_scale,Peak_time
      
        
def Fitting_criteria(lc_g,lc_r):
    time_g,flux_g = lc_g[0], lc_g[1]
    time_r,flux_r = lc_r[0], lc_r[1]
    
    Stage_g, Best_Func_Name_g,best_params_g = Fitting_criteria_single_band(time_g, flux_g)
    Stage_r, Best_Func_Name_r,best_params_r = Fitting_criteria_single_band(time_r, flux_r)
    
    if Stage_g and Stage_r:
        #抽取参数
        
        Rising_scale_g,Fading_scale_g,Peak_time_g = Parameter_extractor(Best_Func_Name_g,best_params_g)
        Rising_scale_r,Fading_scale_r,Peak_time_r = Parameter_extractor(Best_Func_Name_r,best_params_r)
        
        
        if Rising_scale_g is not None and Rising_scale_r is not None:
            error = abs(Rising_scale_g - Rising_scale_r)/max(abs(Rising_scale_g),abs(Rising_scale_r))
            
            if error > 0.8:
                return False
 
        
        if Peak_time_g is not None and Peak_time_r is not None:
            
            error = abs(Peak_time_g - Peak_time_r)
            if error>10:
                return False

        # if Fading_scale_g is not None and Fading_scale_r is not None and (Best_Func_Name_g == Best_Func_Name_r
        #                                                                   or (Best_Func_Name_g =='exp' and Best_Func_Name_r =='gauss+exp')
        #                                                                   or (Best_Func_Name_r =='exp' and Best_Func_Name_g =='gauss+exp')):
        #     error = abs(Fading_scale_g - Fading_scale_r)/max(abs(Fading_scale_g),abs(Fading_scale_r))
        #     if error>0.5:
        #         return False
        # if Fading_scale_g is not None and Fading_scale_r is not None and (Best_Func_Name_g == Best_Func_Name_r
        #                                                                   or (Best_Func_Name_g =='powerlaw' and Best_Func_Name_r =='gauss+powerlaw')
        #                                                                   or (Best_Func_Name_r == 'powerlaw' and Best_Func_Name_g =='gauss+powerlaw')):
            
        #     error = abs(Fading_scale_g - Fading_scale_r)/max(abs(Fading_scale_g),abs(Fading_scale_r))
            
        #     if error>0.5:
        #         return False
            
        
        return True
    else:
        
        return False
    


def simulate_gaussian_lightcurve(t0=10, amp=1.0, sigma=-20.0, baseline=0.0, noise=0.1, size=100):
    """模拟一个高斯光变曲线加上噪声"""
    t = np.linspace(0, 100, size)
    #flux = amp * np.exp((t - t0) / sigma) + baseline
    #flux = amp*((t-0+30)/30)**(-5/3)+baseline
    flux_g = powerlaw_with_c(t,t0=20,tp=-7,alpha=-5/3,amp=20,c=0.01)
    flux_r = powerlaw_with_c(t,t0=30,tp=-1,alpha=-5/3,amp=17,c=0.01)
    flux_g += np.random.normal(0, noise, size)
    flux_r += np.random.normal(0, noise, size)
    return [t, flux_g],[t,flux_r]

if __name__ == "__main__":
    # 模拟双通道光变
    lc_g, lc_r = simulate_gaussian_lightcurve()
    time_g, flux_g = lc_g
    time_r, flux_r = lc_r

    results = {}

    for band, (time, flux) in zip(['g', 'r'], [lc_g, lc_r]):
        print(f"\n=== Fitting for {band}-band ===")
        t0_init = time[np.argmax(flux)]
        f_max = np.max(flux)
        Flag = band

        success, model_name, params, cov, rss, aic, r2, mape = competitive_fitting(
            time=time,
            flux=flux,
            t0_init=t0_init,
            f_max=f_max,
            Flag=Flag
        )

        results[band] = {
            'success': success,
            'model_name': model_name,
            'params': params,
            'rss': rss,
            'aic': aic,
            'r2': r2,
            'mape': mape,
            't0_init': t0_init,
            'f_max': f_max,
            'time': time,
            'flux': flux,
        }

        if success:
            print(f"Best model: {model_name}")
            print(f"AIC: {aic:.4f} | R2: {r2:.2f} | MAPE: {mape:.2f}")
        else:
            print("Fitting failed.")

    # 可视化拟合结果
    ax = Fitting_criteria(lc_g,lc_r)
    if ax:
        plt.figure(figsize=(10, 5))
    
        colors = {'g': 'green', 'r': 'red'}
        linestyles = {'g': '--', 'r': '--'}
        
        for band in ['g', 'r']:
            res = results[band]
            if not res['success']:
                continue
        
            model_name = res['model_name']
            time = res['time']
            flux = res['flux']
            f_max = res['f_max']
            t0_init = res['t0_init']
            Flag = band
        
            # 选择正确模型进行预测
            if model_name == "gauss+exp":
                _, _, _, y_pred = gauss_and_exp_fitting(time, flux, t0_init, Flag=Flag, f_max=f_max)
            elif model_name == "gauss":
                _, _, _, y_pred = Gauss_C_fitting(time, flux, t0_init, f_max=f_max)
            elif model_name == "exp":
                _, _, _, y_pred = exp_fitting(time, flux, t0_init, f_max=f_max)
            elif model_name == 'gauss+powerlaw':
                _, _, _, y_pred = gauss_and_powerlaw_fitting(time, flux)
            elif model_name == 'powerlaw':
                _, _, _, y_pred = powerlaw_fitting(time, flux)
            else:
                y_pred = None
        
            if y_pred is not None:
                # 观测数据点
                plt.plot(time, flux, '.', color=colors[band], label=f'{band}-band Observed')
                # 拟合曲线
                plt.plot(time, y_pred, linestyles[band], color=colors[band], label=f'{band}-band Fit: {model_name}')
        
        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("g and r Band Lightcurve Fitting")
        plt.legend()
        plt.tight_layout()
        plt.show()

    


