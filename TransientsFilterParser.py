import numpy as np
#from WFSTtools.BasicLoggingTools import log_out
#from WFSTtools.Algorithm import Neumann_ratio
import matplotlib.pyplot as plt

from sys import path

path.append('/home/wfstdbadmin/project_WFST/WFST_editable_modules/DIFF_EditableModules/zrf')
path.append('/home/wfstdbadmin/project_WFST/WFST_editable_modules/DIFF_EditableModules/zrf/Filter_open')
path.append('/home/wfstdbadmin/project_WFST/WFST_editable_modules/DIFF_EditableModules/zrf/Mgformer-Filter')
path.append('/Users/astronomy_zrf/Desktop/工作文献2/机器学习/WFST光变曲线读取')
path.append('/Users/astronomy_zrf/Desktop/工作文献2/机器学习/WFST光变曲线读取/Selected_LC/SN_like_from_WFST')


import External_check
import Curve_Fitting_Filter as CFF
import Transformer_ML_Filter as TTC
import warnings
from scipy.optimize import OptimizeWarning

warnings.simplefilter("error", OptimizeWarning)
warnings.filterwarnings("ignore")



FILTER_MAP_NAME='TDE_candidate1' #如果不希望之前 filter得到的 map被覆盖掉/抹除，则可以换个新名称


N_PROCESSOR=25 #使用的并行进程数，建议不要超过30个

# Construct LC

class LightCurve:
    def __init__(self, lc_g=None, lc_r=None, lc_u=None):
        self._data = {}
        if lc_g:
            self._data['g'] = self._wrap_band(lc_g)
        if lc_r:
            self._data['r'] = self._wrap_band(lc_r)
        if lc_u:
            self._data['u'] = self._wrap_band(lc_u)

    class Band:
        def __init__(self, time, flux, flux_err):
            self.time = time
            self.flux = flux
            self.flux_err = flux_err

        def __repr__(self):
            return f"<Band: time={len(self.time)}, flux={len(self.flux)}, flux_err={len(self.flux_err)}>"

    def _wrap_band(self, lc):
        return self.Band(*lc)

    @property
    def g(self):
        return self._data.get('g')

    @property
    def r(self):
        return self._data.get('r')

    @property
    def u(self):
        return self._data.get('u')
    
    def remove_outliers(self, band_keys=("g", "r"), threshold=100):
        """Remove points with abs(flux) > threshold from specified bands."""
        for key in band_keys:
            band = self._data.get(key)
            if band:
                mask = abs(band.flux) <= threshold
                band.time = band.time[mask]
                band.flux = band.flux[mask]
                band.flux_err = band.flux_err[mask]

    def plot(self, bands=("g", "r", "u"), title=None):
            colors = {"g": "green", "r": "red", "u": "purple"}
            plt.figure(figsize=(5, 10))

            for band in bands:
                band_data = getattr(self, band, None)
                if band_data and band_data.time:
                    plt.errorbar(
                        band_data.time,
                        band_data.flux,
                        yerr=band_data.flux_err,
                        fmt='o',
                        label=f'{band}-band',
                        color=colors.get(band, 'black'),
                        alpha=0.7
                    )

            plt.xlabel("MJD")
            plt.ylabel("Flux")
            if title:
                plt.title(title)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


def from_json_to_lc(source_candidate):

    time_g = [];time_r = [];time_u = []
    flux_g = [];flux_r = [];flux_u = []
    flux_err_g = [];flux_err_r = [];flux_err_u = []


    #data_ori = source_candidate['sources']
    Alert_id = list(source_candidate['sources'].keys())
    for alert in Alert_id:
        if len(source_candidate['sources'][alert])>=24:
            if source_candidate['sources'][alert]['band'] == 'g':
                if source_candidate['sources'][alert]['psf_flux']>0 and source_candidate['sources'][alert]['psf_snr']>5:
                    time_g.append(source_candidate['sources'][alert]['mjd'])
                    flux_g.append(source_candidate['sources'][alert]['psf_flux'])
                    flux_err_g.append(source_candidate['sources'][alert]['psf_flux_err'])

            if source_candidate['sources'][alert]['band'] == 'r':
                if source_candidate['sources'][alert]['psf_flux']>0 and source_candidate['sources'][alert]['psf_snr']>5:
                    time_r.append(source_candidate['sources'][alert]['mjd'])
                    flux_r.append(source_candidate['sources'][alert]['psf_flux'])
                    flux_err_r.append(source_candidate['sources'][alert]['psf_flux_err'])

            if source_candidate['sources'][alert]['band'] == 'u':
                if source_candidate['sources'][alert]['psf_flux']>0 and source_candidate['sources'][alert]['psf_snr']>5:
                    time_u.append(source_candidate['sources'][alert]['mjd'])
                    flux_u.append(source_candidate['sources'][alert]['psf_flux'])
                    flux_err_u.append(source_candidate['sources'][alert]['psf_flux_err'])


    lc_g = [time_g,flux_g,flux_err_g]
    lc_r = [time_r,flux_r,flux_err_r]
    lc_u = [time_u,flux_u,flux_err_u]

    lc = LightCurve(lc_g, lc_r, lc_u)

    return lc

def clean_lc(band):
    time = np.array(band.time)
    flux = np.array(band.flux)
    #magnitude = 15
    mask = np.abs(flux) <= 4
    
    return np.array([time[mask], flux[mask]])

def flux_max_check(flux):
    
    if np.max(flux)>0.02:
        return True
    else:
        return False






def sources_filter(candidate_source):
    
    if not External_check.filter_func(candidate_source):
        return False

    source_name = str(candidate_source['objectID'])
    lc_data = from_json_to_lc(candidate_source)

    

    lc_g = clean_lc(lc_data.g)
    lc_r = clean_lc(lc_data.r)
    
    if len(lc_g[0])<=10 or len(lc_r[0])<=10:         
        return False
    
    flux_limit_g = flux_max_check(lc_g[1]);flux_limit_r = flux_max_check(lc_r[1])
    
    if flux_limit_g and flux_limit_r:
        pass
    else:
        return False
    
    if len(lc_g[0])<=10 or len(lc_r[0])<=10:
        return False
    
    
    
    
    

    #if CFF.Fitting_criteria_single_band(lc_g[0], lc_g[1])[0] and CFF.Fitting_criteria_single_band(lc_r[0], lc_r[1])[0]:
    if CFF.Fitting_criteria(lc_g,lc_r):
    #if True:
        Probs = TTC.Get_Classification(
            lc_g, lc_r,
            model_name='/home/wfstdbadmin/project_WFST/WFST_editable_modules/DIFF_EditableModules/zrf/Filter_open/final_model_ZTF.pth',
            model_type='interp_GP'
        )
        
        Probs_np = Probs.to("cpu").numpy()
        TDE_score = Probs_np[0][0]
        Max_score = np.max(Probs_np[0])
        
        if TDE_score>0.0:
        
            print(f"Source ID - {source_name}: ")
            print(TTC.classify_tensor(Probs, verbose=True))
            return True
        else:
            return False

    return False


if __name__ == "__main__":
    import rdlc_WFST
    import numpy as np
    import rdlc_WFST
    import rdlc_ZTF
    import PLAsTICC_res
    import rdlf_WFST_json
    
    LC_typefile = 'WFST'
    data_lc = "/Users/astronomy_zrf/Desktop/工作相关/WFST超新星验证集/疑似SN/WFST_LC_9482087346208995_WFST J104434.56+042812.7.txt"
    
    
    if LC_typefile == 'WFST':
        database_test1 = rdlc_WFST.rd(data_lc)
        database_test1 = rdlc_WFST.mutband_lc(database_test1)
    if LC_typefile == 'ZTF':
        database_test1 = rdlc_ZTF.ZTF_json(data_lc)
        database_test1 = rdlc_ZTF.ZTF_2_WFST_flux(database_test1)

    elif LC_typefile == 'ZTF_lasair':
        database_test1 = rdlc_ZTF.ZTF_json_lasair(data_lc)
        database_test1 = rdlc_ZTF.ZTF_2_WFST_flux(database_test1)
    
    
    processed_data = database_test1
    time_g = processed_data['g']['MJD']
    flux_g = processed_data['g']['psf_flux']
    err_g  = processed_data['g']['psf_flux_err']

    time_r = processed_data['r']['MJD']
    flux_r = processed_data['r']['psf_flux']
    err_r  = processed_data['r']['psf_flux_err']
    
    lc_g = [time_g,flux_g]
    lc_r = [time_r,flux_r]
    
    A1 = CFF.Fitting_criteria(lc_g,lc_r)
    # A2 = TTC.Get_Classification(
    #     lc_g, lc_r,
    #     model_name='/home/wfstdbadmin/project_WFST/WFST_editable_modules/DIFF_EditableModules/zrf/Filter_open/final_model_ZTF.pth',
    #     model_type='interp_GP'
    # )
    
    print(A1)
    #print(A2)
    
        
    
    



