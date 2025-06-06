#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:16:37 2024

@author: astronomy_zrf
关于处理单个ZTF光变曲线的函数都写在这里
包括以后对于机器学习需要的预处理函数也写在这里

之前确实有个包的用法是这样的：
from ztfquery import lightcurve
lcq = lightcurve.LCQuery.from_position(197.501495, +75.721959, 5)
或者
lcq1 = lightcurve.LCQuery.from_id([''])但是ID里面只能输入具体的编号，不能输入source name

lcq.show()

想要实现这个方法呢，需要登录irsa
账号已经注册了，安全性起见就不写在这里了
"""

def ZTF_pos_lc(RA,DEC,radius = 5):
    #这个算法用于绕过Lasir直接通过API获取ZTF光变曲线数据
    #需要赤经，赤纬，光圈半径
    from ztfquery import lightcurve
    lcq = lightcurve.LCQuery.from_position(RA,DEC,radius)

    data = lcq.data
    return data



import json

    


#from ztfquery import lightcurve
#lcq = lightcurve.LCQuery(file_data)



#============================================
#抽取不同波段的光变曲线，独立成数列和时间。
#存储格式：字典嵌套，[{源编号:x,时间:[],u:[],g:[],r:[],i:[],z:}{...}]
#============================================

def m2f(mag):
    f = 10 ** (0.4 * (22.5 - float(mag)))
    return f
def merr2f(mag,mag_err):
    #print('mag_err = ',mag_err)
    f_err = mag_err/1.0857*m2f(mag)
    return f_err


def ZTF_json(fp,out_put_json = 0):
    #============================================
    #ZTF中的日期也是存在谬误的，需要调整
    #抽取不同波段的光变曲线，独立成数列和时间。
    #存储格式：字典嵌套，[{有效源编号:x0,最终源编号:x1,时间:[],u:[],g:[],r:[],i:[],z:[]...time:[]]}{...}
    #可以选择输出为json,默认不输出
    #data输入格式为json
    #ZTF疑似只有两个波段，但是保险起见还是安装5个波段在里面
    #数据均以星等方式存储
    #也是只能处理单个光变曲线,
    #============================================
    
    
    with open(fp, 'r') as fcc_file:
        data = json.load(fcc_file)
    
    target_dict = []
    #source_count = len(data['models'][0]['realizations'])
    lc_length = len(data)
    
    name = fp.split('/')

    
    for i in range(len(name)):
        if len(name[i])>5:
            
            if name[i][0:3] == 'ZTF' and name[i][-1]=='n':
                name1 = name[i]
                break
    
    name2 = name1.split('_')
    name = name2[0]

    
    
    target_dict.append({'u':[],'g':[],'r':[],'i':[],'z':[]
                        ,'ue':[],'ge':[],'re':[],'ie':[],'ze':[],'time_MJD_u':[],'time_MJD_g':[],'time_MJD_r':[]
                        ,'time_MJD_i':[],'time_MJD_z':[],'ul':[],'gl':[],'rl':[],'il':[],'zl':[],'tul':[],'tgl':[]
                        ,'trl':[],'til':[],'tzl':[],'name':name})
    
    for i in range(lc_length):
        #print(data[i])
        if data[i]['unforced_mag_status'] != 'limit':
            if data[i]['filter'] == 'u':
           
                
               target_dict[0]['u'].append(float(data[i]['unforced_mag']))
               target_dict[0]['ue'].append(float(data[i]['unforced_mag_error']))
               target_dict[0]['time_MJD_u'].append(float(data[i]['MJD']))
                    
            if data[i]['filter'] == 'g':
                target_dict[0]['g'].append(float(data[i]['unforced_mag']))
                target_dict[0]['ge'].append(float(data[i]['unforced_mag_error']))
                target_dict[0]['time_MJD_g'].append(float(data[i]['MJD']))
                    
            
            if data[i]['filter'] == 'r':
                target_dict[0]['r'].append(float(data[i]['unforced_mag']))
                target_dict[0]['re'].append(float(data[i]['unforced_mag_error']))
                target_dict[0]['time_MJD_r'].append(float(data[i]['MJD']))
                
            if data[i]['filter'] == 'i':
                target_dict[0]['i'].append(float(data[i]['unforced_mag']))
                target_dict[0]['ie'].append(float(data[i]['unforced_mag_error']))
                target_dict[0]['time_MJD_i'].append(float(data[i]['MJD']))
    
            if data[i]['filter'] == 'z':
                target_dict[0]['z'].append(float(data[i]['unforced_mag']))
                target_dict[0]['ze'].append(float(data[i]['unforced_mag_error']))
                target_dict[0]['time_MJD_z'].append(float(data[i]['MJD']))
                
        if data[i]['unforced_mag_status'] == 'limit':
            #只有下界的情况，是得不到测光误差的
            #但是，这个时候为了方便后续画图，不能直接将下界的情况拼接
            #它会输出下限点，单独存在ul，tul类似的列表中，与正常的测光数据区分开
            if data[i]['filter'] == 'u':
           
                
               target_dict[0]['ul'].append(float(data[i]['unforced_mag']))
             
               target_dict[0]['tul'].append(float(data[i]['MJD']))
                    
            if data[i]['filter'] == 'g':
                target_dict[0]['gl'].append(float(data[i]['unforced_mag']))
               
                target_dict[0]['tgl'].append(float(data[i]['MJD']))
                    
            
            if data[i]['filter'] == 'r':
                target_dict[0]['rl'].append(float(data[i]['unforced_mag']))
           
                target_dict[0]['trl'].append(float(data[i]['MJD']))
                
            if data[i]['filter'] == 'i':
                target_dict[0]['il'].append(float(data[i]['unforced_mag']))
           
                target_dict[0]['til'].append(float(data[i]['MJD']))
    
            if data[i]['filter'] == 'z':
                target_dict[0]['zl'].append(float(data[i]['unforced_mag']))
         
                target_dict[0]['tzl'].append(float(data[i]['MJD']))
        
        
        if out_put_json != 0:
            with open('multi_lc.json', 'w') as file:
                json.dump(target_dict, file)
                     
        
    
    return target_dict


def trans_f1_2_f2(f1,zp):
    import numpy as np
    f2 = 3631000*10**(-(zp-2.5*np.log10(f1))/2.5)
    return f2

def trans_fe1_2_fe2(f_diffunc,f_diff,f_ab):
    #f = flux_AB   fe = forced_diffunc  ft = forced_diffflux
    return f_ab*f_diffunc/f_diff

def ZTF_json_lasair(fp,out_put_json = 0):
    #============================================
    #ZTF中的日期也是存在谬误的，需要调整
    #抽取不同波段的光变曲线，独立成数列和时间。
    #存储格式：字典嵌套，[{有效源编号:x0,最终源编号:x1,时间:[],u:[],g:[],r:[],i:[],z:[]...time:[]]}{...}
    #可以选择输出为json,默认不输出
    #data输入格式为json
    #ZTF疑似只有两个波段，但是保险起见还是安装5个波段在里面
    #数据均以星等方式存储
    #也是只能处理单个光变曲线,
    #============================================
    
    
    with open(fp, 'r') as fcc_file:
        data_all = json.load(fcc_file)
    
    target_dict = []
    #source_count = len(data['models'][0]['realizations'])
    lc_length_candidate = len(data_all['candidates'])
    
    if 'forcedphot' in data_all:
        lc_length_forced = len(data_all['forcedphot'])
    
    name = fp.split('/')

    
    for i in range(len(name)):
        if len(name[i])>5:
            
            if name[i][0:3] == 'ZTF' and name[i][-1]=='n':
                name1 = name[i]
                break
    
    name2 = name1.split('_')
    name = name2[0]
    
    data = data_all['candidates']
    #data_forced = data_all['forcedphot']
    
    target_dict.append({'u':[],'g':[],'r':[],'i':[],'z':[]
                        ,'ue':[],'ge':[],'re':[],'ie':[],'ze':[],'time_MJD_u':[],'time_MJD_g':[],'time_MJD_r':[]
                        ,'time_MJD_i':[],'time_MJD_z':[],'ul':[],'gl':[],'rl':[],'il':[],'zl':[],'tul':[],'tgl':[]
                        ,'trl':[],'til':[],'tzl':[],'name':name,'forced_gt':[],'forced_gf':[],'forced_gfe':[]
                        ,'forced_rt':[],'forced_rf':[],'forced_rfe':[],'gzp':[],'rzp':[]})
    
    for i in range(lc_length_candidate):
        #print(data[i])
        if "magpsf" in data[i] and data[i]["isdiffpos"]!="f":
            
            if data[i]['fid'] == 1:
                
               target_dict[0]['g'].append(float(data[i]['magpsf']))
               target_dict[0]['ge'].append(float(data[i]['sigmapsf']))
               #JD --> MJD
               target_dict[0]['time_MJD_g'].append(float(data[i]['jd'])-2400000.5)
                    

    
            if data[i]['fid'] == 2 and data[i]["isdiffpos"]!="f":
                #fid = 2 --> rband
                target_dict[0]['r'].append(float(data[i]['magpsf']))
                target_dict[0]['re'].append(float(data[i]['sigmapsf']))
                #jd --> MJD
                target_dict[0]['time_MJD_r'].append(float(data[i]['jd'])-2400000.5)
        elif 'diffmaglim' in data[i]:
            if data[i]['fid'] == 1:
                #fid = 1 --> gband
                target_dict[0]['gl'].append(float(data[i]['diffmaglim']))
           
                target_dict[0]['tgl'].append(float(data[i]['jd'])-2400000.5)
    
            if data[i]['fid'] ==2:
                #fid = 2 --> rband
                target_dict[0]['rl'].append(float(data[i]['diffmaglim']))
         
                target_dict[0]['trl'].append(float(data[i]['jd'])-2400000.5)
                
    if 'forcedphot' in data_all:    
        data = data_all["forcedphot"]
        
        for i in range(lc_length_forced):
            
               
            zp = float(data[i]['magzpsci'])
            ft = data[i]["forcediffimflux"]
            f = trans_f1_2_f2(ft,zp)
            fe = data[i]['forcediffimfluxunc']
            fe_fixed = trans_fe1_2_fe2(fe, ft, f)
            
            
            
            
            
            
            if  "forcediffimflux" in data[i] and data[i]["forcediffimflux"]!=-99999.0 and data[i]["forcediffimflux"]/data[i]['forcediffimfluxunc']>5:
                data[i]["forcediffimflux"] = f
                data[i]['forcediffimfluxunc'] = fe_fixed
                if data[i]['fid'] == 1:
                    #fid = 1 --> gband
                    #
                    
                   target_dict[0]['forced_gf'].append(float(data[i]['forcediffimflux']))
                   target_dict[0]['forced_gfe'].append(float(data[i]['forcediffimfluxunc']))
                   target_dict[0]['gzp'].append(float(data[i]['magzpsci']))
                   #JD --> MJD
                   target_dict[0]['forced_gt'].append(float(data[i]['jd'])-2400000.5)
                if data[i]['fid'] == 2:
                    #fid = 1 --> gband
                    
                   target_dict[0]['forced_rf'].append(float(data[i]['forcediffimflux']))
                   target_dict[0]['forced_rfe'].append(float(data[i]['forcediffimfluxunc']))
                   target_dict[0]['rzp'].append(float(data[i]['magzpsci']))
                   #JD --> MJD
                   target_dict[0]['forced_rt'].append(float(data[i]['jd'])-2400000.5)
                    
        
    if out_put_json != 0:
        with open('multi_lc.json', 'w') as file:
            json.dump(target_dict, file)
                     
        
    
    return target_dict




#with open('ZTF22abajudi_difference_photometry.json','r') as file:
#    file_data = json.load(file)
    #其实这里已经直接得到了测光数据曲线。


def ZTF_splot(data_from_rdlc):
    import matplotlib.pyplot as plt#reload
    data_lc = data_from_rdlc
    fig = plt.figure()

    ax = fig.add_subplot()



    for band in ['u','g','r','i','z']:
        if data_lc[band]!=[]:
            mag = data_lc[band]
            t = data_lc['time_MJD_'+band]
            err = data_lc[band+'e']
            l_mag = data_lc[band+'l']
            l_t = data_lc['t'+band+'l']
            ax.errorbar(t,mag,err,fmt='.',label = band)
            ax.plot(l_t,l_mag,'v',label = band+' lower limit')

    ax.legend()
    ax.set_xlabel('MJD')
    ax.set_ylabel('magnitude(ZTF)')
    ax.invert_yaxis()


def ZTF_2_WFST(target_data):
    #将ZTF的数据格式写成跟WFST一样的格式，顺带排个序
    #只测出下限的点暂时不参与分类
    #这些下限点该怎么处理还是不清楚
    #ZTF里面还有著名的nan情况
    final_data = {'u':{'MJD':[],'psf_mag':[],'psf_mag_err':[]},
                  'g':{'MJD':[],'psf_mag':[],'psf_mag_err':[]},
                  'r':{'MJD':[],'psf_mag':[],'psf_mag_err':[]},
                  'i':{'MJD':[],'psf_mag':[],'psf_mag_err':[]},
                  'z':{'MJD':[],'psf_mag':[],'psf_mag_err':[]}}
    #print(target_data)
    
    #print(target_data['time_MJD_u'])
    
    target_data = target_data[0]
    
    
    g = zip(target_data['time_MJD_g'],target_data['g'],target_data['ge'])
    if target_data['time_MJD_g']!=[]:
        g = sorted(g)
        final_data['g']['MJD'],final_data['g']['psf_mag'],final_data['g']['psf_mag_err'] = zip(*g)
    
    r = zip(target_data['time_MJD_r'],target_data['r'],target_data['re'])
    if target_data['time_MJD_r']!=[]:
        r = sorted(r)
        final_data['r']['MJD'],final_data['r']['psf_mag'],final_data['r']['psf_mag_err'] = zip(*r)
    
    
   
    
    return final_data

def ZTF_2_WFST_forcephot(target_data):
    #将ZTF的数据格式写成跟WFST一样的格式，顺带排个序
    #只测出下限的点暂时不参与分类
    #这些下限点该怎么处理还是不清楚
    #ZTF里面还有著名的nan情况
    final_data = {'u':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'g':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'r':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'i':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'z':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'objectID':target_data[0]['name']}
    #print(target_data)
    
    #print(target_data['time_MJD_u'])
    
    

    
    target_data = target_data[0]
    
    for i in range(len(target_data['forced_gt'])):
        #这个顺序不能调
        zp = target_data['gzp'][i]

        f1 = target_data['forced_gf'][i]
        target_data['forced_gf'][i] =  f1
        
        f1 = target_data['forced_gfe'][i]
        target_data['forced_gfe'][i] = f1
        
       
        
        
    
    for i in range(len(target_data['forced_rt'])):
        zp = target_data['rzp'][i]
        f1 = target_data['forced_rfe'][i]
        target_data['forced_rfe'][i] = f1
        f1 = target_data['forced_rf'][i]
        target_data['forced_rf'][i] = f1
        
        
    
    g = zip(target_data['forced_gt'],target_data['forced_gf'],target_data['forced_gfe'])
    if target_data['forced_gt']!=[]:
        g = sorted(g)
        final_data['g']['MJD'],final_data['g']['psf_flux'],final_data['g']['psf_flux_err'] = zip(*g)
    
    r = zip(target_data['forced_rt'],target_data['forced_rf'],target_data['forced_rfe'])
    if target_data['forced_rt']!=[]:
        r = sorted(r)
        final_data['r']['MJD'],final_data['r']['psf_flux'],final_data['r']['psf_flux_err'] = zip(*r)
    
    
   
    #print(final_data)
    return final_data




def ZTF_2_WFST_flux(target_data):
    #将ZTF的数据格式写成跟WFST一样的格式，顺带排个序
    #只测出下限的点暂时不参与分类
    #这些下限点该怎么处理还是不清楚
    #ZTF里面还有著名的nan情况
    final_data = {'u':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'g':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'r':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'i':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'z':{'MJD':[],'psf_flux':[],'psf_flux_err':[]},
                  'objectID':target_data[0]['name']}
    #print(target_data)
    
    #print(target_data['time_MJD_u'])
    
    target_data = target_data[0]
    
    for i in range(len(target_data['time_MJD_g'])):
        #这个顺序不能调
        target_data['ge'][i] = merr2f(target_data['g'][i],target_data['ge'][i])
        target_data['g'][i] = m2f(target_data['g'][i])
        
       
        
        
    
    for i in range(len(target_data['time_MJD_r'])):
        target_data['re'][i] = merr2f(target_data['r'][i],target_data['re'][i])
        target_data['r'][i] = m2f(target_data['r'][i])
        
        
    
    g = zip(target_data['time_MJD_g'],target_data['g'],target_data['ge'])
    if target_data['time_MJD_g']!=[]:
        g = sorted(g)
        final_data['g']['MJD'],final_data['g']['psf_flux'],final_data['g']['psf_flux_err'] = zip(*g)
    
    r = zip(target_data['time_MJD_r'],target_data['r'],target_data['re'])
    if target_data['time_MJD_r']!=[]:
        r = sorted(r)
        final_data['r']['MJD'],final_data['r']['psf_flux'],final_data['r']['psf_flux_err'] = zip(*r)
    
    
   
    
    return final_data








def lc_mean(lc_data,mean_day=2):
    import numpy as np
    #该函数用来预处理光变曲线，将一天之内的测光取平均值输出
    #这玩意我记得我以前写过，我去翻一下
    #没翻到，重新写吧
    #今天第一次发现一个非常重量级的事情，那就是实际上，WFST光变曲线的那些个数据
    #居然是没有经过排序的，甚至是时间乱序
    #排序内容写在rdlc_WFST中的rd了
    #这个涉及具体的一个分类工作，就暂时不考虑误差了，默认平滑尺度2d
    #只不过现在的问题是，我是因为平均流量呢，还是平均星等呢……
    t_g = lc_data['g']['MJD']
    t_r = lc_data['r']['MJD']
    mag_r = lc_data['r']['psf_mag']
    mag_g = lc_data['g']['psf_mag']
    
    t_gl = []
    t_rl = []
    mag_rl = []
    mag_gl = []
    
    bins_g = [[]]#index bin 0 = 0
    bins_mag_g = [[]]
    
    bins_r =[[]]
    bins_mag_r = [[]]
    
    index_bin_g = 0
    index_bin_r = 0
    
    
    
    for i in range(len(t_g)):
        if i <len(t_g)-1:
            if (t_g[i+1] - t_g[i] <=mean_day) and str(t_g[i])!='nan' :#2天之内的数据点归于一个bin，具体天数可以自己填，默认2天
                bins_g[index_bin_g].append(t_g[i])
                bins_mag_g[index_bin_g].append(mag_g[i])
            else:
                index_bin_g +=1
                bins_g.append([])
                bins_mag_g.append([])
            
                bins_g[index_bin_g].append(t_g[i+1])
                bins_mag_g[index_bin_g].append(mag_g[i])
                
    for i in range(len(t_r)):
        if i <len(t_r)-1:
            if (t_r[i+1] - t_r[i] <=mean_day) and str(t_r[i])!='nan':#1天之内的数据点归于一个bin
                bins_r[index_bin_r].append(t_r[i])
                bins_mag_r[index_bin_r].append(mag_r[i])
            else:
                index_bin_r +=1
                bins_r.append([])
                bins_mag_r.append([])
            
                bins_r[index_bin_r].append(t_r[i+1])
                bins_mag_r[index_bin_r].append(mag_r[i])
                
    
    for i in range(len(bins_g)):
        t_gl.append(np.mean(bins_g[i]))
    
    for i in range(len(bins_r)):
        t_rl.append(np.mean(bins_r[i]))
        
    for i in range(len(bins_mag_g)):
        mag_gl.append(np.mean(bins_mag_g[i]))

    for i in range(len(bins_mag_r)):
        mag_rl.append(np.mean(bins_mag_r[i]))
    
    lc_data['g']['MJD'] = t_gl;lc_data['r']['MJD'] = t_rl
    lc_data['g']['psf_mag'] = mag_gl;lc_data['r']['psf_mag'] = mag_rl
    
    return lc_data


    