from pandas.core.computation.check import NUMEXPR_INSTALLED
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import h5py
from time import time
from tqdm import tqdm
import copy
from random import randint

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost

from sklearn.gaussian_process.kernels import ConstantKernel as CK
from sklearn.gaussian_process.kernels import RBF

from sklearn.model_selection import GridSearchCV


import sys
# print(sys.getrecursionlimit())
# sys.setrecursionlimit(8000) # Voting regressor


###################################################################################################################
# Global Constants

jupiter_radius = 7.1492e7
jupiter_mass   = 1.89813e27
solar_radius   = 6.95700e8
solar_mass     = 1.98847e30


###################################################################################################################
# Metrics

def score(y_true,y_pred):
    score = 100*(10-np.sqrt(np.sum((y_pred/y_true-1)**2)/(y_true.shape[0])/6))
    return score

def mare(y_true,y_pred):
    # y_pred[y_pred<=0] = 1e-9
    score = np.mean(np.abs( (y_pred-y_true)/y_true ))
    return score

# Better metric?
def mare_per_col(y_true,y_pred):
    # y_pred[y_pred<=0] = 1e-9
    #score = np.mean(np.abs((y_pred-y_true)/y_true),axis=0)
    # quant = np.quantile(np.abs((y_pred[:]-y_true[:])/y_true[:]),[0.25,0.75], axis=0) # quant not mare anymore
    # score = (quant[1,:]-quant[0,:]).reshape(1,6)
    # score = np.mean(np.abs( (y_pred-y_true)/y_true ),axis=0).reshape(1,6)
    score = np.mean(np.abs((y_true-y_pred)/y_true), axis=0)
    return score

# SK-Learn score: Coefficient of determination
def sklearn_score(y_true,y_pred):
    return  1. - np.mean(np.sum((y_true-y_pred)**2,axis=0)/np.sum((y_true-np.mean(y_true,axis=0))**2,axis=0))

###################################################################################################################
# Transforms

def standardize(x):
    means = np.mean(x,axis=0)
    stds = np.std(x,axis=0)
    x =(x-means)/stds
    return x, means, stds

def normalize(x):
    means = np.mean(x,axis=1)
    stds = np.std(x,axis=1)
    x = (x-means[:,None])/stds[:,None]
    return x, means, stds


def transform_data( input1, #spectra
                    input2, #aux data
                    target, #fm data
                    train_test_total_samples,
                    train_test_split_test_percentage,
                    train_test_split_random_state = 0,
                    input1_transform = 'standardize',
                    input2_transform = 'standardize',
                    target_transform = 'standardize',
                    noise_ppm = None,
                    target_ppm_concentrations = False,
                    include_input2 = False,
                    include_input1_mean_std = False, # only applicable for the input transform being normalize
                    ):

    if noise_ppm is not None:
        input1 = generate_noisy_spectra(input1,noise_ppm)
    if target_ppm_concentrations:
        target[:,1:] = np.power(10,target[:,1:])

    # Original
    if input1_transform=='standardize':
        input1, input1_means, input1_stds = standardize(input1)
    elif input1_transform=='normalize':
        input1, input1_means, input1_stds = normalize(input1)

    # Auxiliary Data
    if input2_transform=='standardize':
        input2_train, input2_means, input2_stds = standardize(input2)
    elif input2_transform=='normalize':
        input2, input2_means, input2_stds = normalize(input2)

    # Target Data
    if target_transform=='standardize':
        y, y_means, y_stds = standardize(target)
    elif target_transform=='normalize':
        y, y_means, y_stds = normalize(target)

    # Set input x for model
    if input1_transform == 'standardize':
        if include_input2: 
            x = np.hstack([input1,input2])
        else: 
            x = copy.copy(input1)

    elif input1_transform == 'normalize':
        if include_input2 and include_input1_mean_std: 
            x = np.hstack([input1, input1_means[:,None], input1_stds[:,None], input2])
        elif include_input2 and not include_input1_mean_std: 
            x = np.hstack([input1, input2])
        elif not include_input2 and include_input1_mean_std: 
            x = np.hstack([input1, input1_means[:,None], input1_stds[:,None]])
        else:
            x = copy.copy(input1)

    total_sample_size = x.shape[0]
    np.random.seed(0)
    selected_inds = np.random.choice(np.arange(total_sample_size), 
                                     size = train_test_total_samples, 
                                     replace = False)

    x_train, x_test, y_train, y_test = train_test_split(x[selected_inds],
                                                        y[selected_inds],
                                                        test_size=train_test_split_test_percentage,
                                                        random_state=train_test_split_random_state)

    return x_train, x_test, y_train, y_test, y_means, y_stds

def create_true_bin_and_bias_arrays(y_true,y_pred):
    inds_all = np.arange(0,y_true.shape[0])
    inds_split = []
    partition = 23
    samples_per_bin = y_true.shape[0]//partition
    for i in range(partition):
        if i<partition-1:
            inds_split.append(inds_all[i*samples_per_bin:(i+1)*samples_per_bin])
        else:
            inds_split.append(inds_all[i*samples_per_bin:])

    true_bins = np.zeros((partition,y_true.shape[1]),dtype = np.float64 )
    args_sorted = np.argsort(y_true,axis=0)
    true_sorted = np.sort(y_true,axis=0)

    avg_truebin_deviation = np.zeros((partition,y_true.shape[1]),dtype = np.float64 )
    std_pred_truebins = np.zeros((partition,y_true.shape[1]),dtype = np.float64 )
    pred_sortedbytrue = np.zeros((y_true.shape[0],y_true.shape[1]),dtype = np.float64 )

    for i in range(y_true.shape[1]):
        pred_sortedbytrue[:,i] = y_pred[args_sorted[:,i],i]
        
    for i,group in enumerate(inds_split):
        true_bins[i,:] = np.mean(true_sorted[group],axis=0)
        avg_truebin_deviation[i,:] = np.mean(np.abs(true_sorted[group]-pred_sortedbytrue[group]),axis=0)
        std_pred_truebins[i,:] = np.std(np.abs(true_sorted[group]-pred_sortedbytrue[group]),axis=0)

    return true_bins, avg_truebin_deviation, std_pred_truebins


def create_pred_bin_and_error_arrays(y_true,y_pred):
    inds_all = np.arange(0,y_true.shape[0])
    inds_split = []
    partition = 23
    samples_per_bin = y_true.shape[0]//partition
    for i in range(partition):
        if i<partition-1:
            inds_split.append(inds_all[i*samples_per_bin:(i+1)*samples_per_bin])
        else:
            inds_split.append(inds_all[i*samples_per_bin:])

    args_pred_sorted = np.argsort(y_pred,axis=0)
    pred_sorted = np.sort(y_pred,axis=0)
    pred_bins = np.zeros((partition,y_true.shape[1]), dtype = np.float64 )

    avg_predbin_deviation = np.zeros((partition,y_true.shape[1]),dtype = np.float64 )
    std_pred_predbins = np.zeros((partition,y_true.shape[1]),dtype = np.float64 )
    true_sortedbypred = np.zeros((y_true.shape[0],y_true.shape[1]),dtype = np.float64 )

    for i in range(y_true.shape[1]):
        true_sortedbypred[:,i] = y_true[args_pred_sorted[:,i],i]
        
    for i,group in enumerate(inds_split):
        pred_bins[i,:] = np.mean(pred_sorted[group],axis=0)
        avg_predbin_deviation[i,:] = np.mean(np.abs(pred_sorted[group]-true_sortedbypred[group]),axis=0)
        std_pred_predbins[i,:] = np.std(np.abs(pred_sorted[group]-true_sortedbypred[group]),axis=0)

    return pred_bins, avg_predbin_deviation, std_pred_predbins



###################################################################################################################
# Training

def supervised_training(x,
                        y, 
                        x_test,
                        y_test,
                        target_means,
                        target_stds,
                        method,
                        y_keys,
                        method_parameters = None,
                        ensemble_regressors = None, # SVM, KNN, DT, RF (Voting), SVM, KNN, DT (Stacking) were used in the paper
                        ensemble_regressor_parameters = None,
                        final_regressor = None, # RF (Stacking only) were used in the paper
                        final_regressor_parameters = None,
                        target_transform = 'standardize',
                        target_ppm_concentrations = False,
                        bootstrap = False
                        ):
    
    np.random.seed(0)
    
    pls_default  = PLSRegression()
    svm_default  = SVR()
    knn_default  = KNeighborsRegressor()
    rnn_default  = RadiusNeighborsRegressor()
    dt_default   = DecisionTreeRegressor()
    rf_default   = RandomForestRegressor(n_estimators=10)
    xgb_default  = xgboost.XGBRegressor(random_state=42)

    
    pls  = PLSRegression
    svm  = SVR
    knn  = KNeighborsRegressor
    rnn  = RadiusNeighborsRegressor
    dt   = DecisionTreeRegressor
    rf   = RandomForestRegressor
    xgb  = xgboost.XGBRegressor

    # Dictionaries of regressors
    default_methods_dict = {'PLS': pls_default,
                            'SVM': svm_default,
                            'KNN': knn_default,
                            'RNN': rnn_default,
                            'DT': dt_default,
                            'RF': rf_default,
                            'XGB': xgb_default}

    methods_dict = {'PLS': pls,
                    'SVM': svm,
                    'KNN': knn,
                    'RNN': rnn,
                    'DT': dt,
                    'RF': rf,
                    'XGB': xgb}

    #  Initialize your regressor
    if method in list(methods_dict.keys()):
        regressor = methods_dict[method](**method_parameters)

    elif method=='VOTE':
        pass
    elif method=='STACK':
        pass
    elif method=='XGB':
        regressor = xgb
    else:
        print('Warning: Invalid method selection.')
    
    trained_regressor = []
    # svm_track = False
    # if method == 'SVM':
    #     svm_track = True
    # elif ensemble_regressors is not None:
    #     if 'SVM' in ensemble_regressors:
    #         svm_track = True
    #     elif final_regressor=='SVM':
    #         svm_track = True
    
    if (method=='SVM') or ensemble_regressors is not None:
        scores = np.empty((6,1))
        pred   = np.empty((y_test.shape[0],6))
        duration = 0.
        for i in range(y.shape[1]):
            if method=='SVM':
                regressor = SVR(**method_parameters)
            elif method=='VOTE':
                estimators = []
                for k,reg_string in enumerate(ensemble_regressors):
                    estimators.append( (reg_string,methods_dict[reg_string](**ensemble_regressor_parameters[k]) ) )
                    # if reg_string=='PLS':
                    #     methods_dict[reg_string](**ensemble_regressor_parameters[k]).predict = lambda x: methods_dict[reg_string](**ensemble_regressor_parameters[k]).predict(x).reshape(x.shape[0],)
                regressor = VotingRegressor(estimators=estimators)
            elif method=='STACK':
                estimators = []
                for k,reg_string in enumerate(ensemble_regressors):
                    estimators.append( (reg_string,methods_dict[reg_string](**ensemble_regressor_parameters[k]) ) )
                if final_regressor_parameters is not None:
                    regressor = StackingRegressor(estimators=estimators,final_estimator=methods_dict[final_regressor](**final_regressor_parameters) )
                else:
                    regressor = StackingRegressor(estimators=estimators,final_estimator=default_methods_dict[final_regressor])
            start = time()
            regressor.fit(x,y[:,i])
            end = time()
            duration+=end-start
            pred[:,i] = regressor.predict(x_test)
            if not bootstrap:
                scores[i] = regressor.score(x_test,y_test[:,i])
                print('Accuracy for '+y_keys[i]+': ',np.round(scores[i],6),'\n')
            trained_regressor.append(regressor)
        if not bootstrap:
            print(method+' Overall Accuracy (sklean score): ',sklearn_score(y_test,pred))
            print(method+' Training Duration: ',duration,' s')
            
    else:
        if method=='XGB':
            start = time()
            regressor.fit(X=x, 
                          y=y, 
                          eval_set=[(x_test, y_test)], 
                          verbose=False  )
            end = time()
        else:
            start = time()
            regressor.fit(x,y)
            end = time()

        pred = regressor.predict(x_test)
        if not bootstrap:
            if method=='XGB':
                score = sklearn_score(y_test,pred)
            else:
                score = regressor.score(x_test,y_test)
            print(method+' Accuracy (sklean score): ',score)
            print(method+' Training Duration: ',end-start,' s')
        trained_regressor.append(regressor)

    if target_transform=='standardize':
        predictions = pred*target_stds+target_means
        y_test_back = y_test*target_stds+target_means
    elif target_transform=='normalize':
        predictions = pred*target_stds[:,None]+target_means[:,None]
        y_test_back = y_test*target_stds[:,None]+target_means[:,None]

    if target_ppm_concentrations:
        predictions[:,1:] = np.where(predictions[:,1:]>1e-12,predictions[:,1:],1e-09)
        predictions[:,1:] = np.log10(predictions[:,1:])
        y_test_back[:,1:] = np.log10(y_test_back[:,1:])

    if not bootstrap:
        print(method+' Mean Absolute Error: ', mare(y_test_back,predictions))

    return trained_regressor, predictions, y_test_back



def bootstrap_training(x,
                        y, 
                        x_test,
                        y_test,
                        single_test_planet_idx,
                        target_means,
                        target_stds,
                        method,
                        y_keys,
                        method_parameters = None,
                        ensemble_regressors = None, # SVM, KNN, DT, RF (Voting), SVM, KNN, DT (Stacking) were used in the paper
                        ensemble_regressor_parameters = None,
                        final_regressor = None, # RF (Stacking only) were used in the paper
                        final_regressor_parameters = None,
                        target_transform = 'standardize',
                        target_ppm_concentrations = False,
                        trials = 100,
                        percentage_of_training_data = 0.8
                        ):

    M=trials
    x_test = x_test[single_test_planet_idx].reshape(1,x_test.shape[1])
    y_test = y_test[single_test_planet_idx].reshape(1,y_test.shape[1])
    trials = np.zeros((M,y_test.shape[1]),dtype=np.float64)
    for i in tqdm(range(M)):
        np.random.seed(i*5)
        selected_inds = np.random.choice(np.arange(x.shape[0]), size = int(x.shape[0]*percentage_of_training_data), replace = False)
        np.random.seed(0)
        trained_regressor_list, predictions, true_test_targets = supervised_training(x = x[selected_inds],
                                                                                    y = y[selected_inds], 
                                                                                    x_test = x_test,
                                                                                    y_test = y_test,
                                                                                    target_means = target_means,
                                                                                    target_stds = target_stds,
                                                                                    method = method,
                                                                                    y_keys = y_keys,
                                                                                    method_parameters = method_parameters,
                                                                                    ensemble_regressors = ensemble_regressors, # SVM, KNN, DT, RF (Voting), SVM, KNN, DT (Stacking) were used in the paper
                                                                                    ensemble_regressor_parameters =  ensemble_regressor_parameters,
                                                                                    final_regressor = final_regressor, # RF (Stacking only) were used in the paper
                                                                                    final_regressor_parameters = final_regressor_parameters,
                                                                                    target_transform = target_transform,
                                                                                    target_ppm_concentrations = target_ppm_concentrations,
                                                                                    bootstrap = True
                                                                                    )
            
        trials[i,:] = predictions.reshape(1,y.shape[1])
    return trials, true_test_targets


###################################################################################################################
# Plotting

def plot_spectra(wavelengths,spectra):
    fig = plt.figure(figsize=(5,4))
    plt.plot(wavelengths, spectra,color='b',lw=4)
    plt.xlabel(r'$\lambda (\mu m)$',size=20)
    plt.ylabel('M',size=20)
    plt.xticks(size=12)
    plt.yticks(size=12)


def generate_noisy_spectra(spectra,noise_ppm):
    np.random.seed(42)
    errors_ppm = noise_ppm*1e-6*np.ones(52)
    spectra_noise_ppm = np.random.normal(spectra, errors_ppm) # Add Gaissian noise to spectra
    spectra_noise_ppm = np.where(spectra_noise_ppm >= 0, spectra_noise_ppm, 0) # Change negative spectra values to zero
    return spectra_noise_ppm


def plot_max_spectra_distribution(spectra):
    spectra_max = np.max(spectra,axis=1)
    fig = plt.figure(figsize=(3,2))
    plt.hist(np.log10(spectra_max), bins=100, color='b')
    # plt.vlines(np.log10(1e-3),0.1,5000,color='r')
    plt.xlabel('Log $M_{max}$', fontsize=12)
    plt.yscale('log')


def plot_auxiliary_distributions(auxiliary_data, auxiliary_keys):
    fig = plt.subplots(3,3,figsize = (18,15),constrained_layout=True)
    for i in range(auxiliary_data.shape[1]):
        plt.subplot(3,3,i+1)
        if i in [0,4,5,6,8]:
            if i!=4:
                plt.hist(np.log10(auxiliary_data[:,i]),color='b',bins=40)
            # plt.hist(np.log10(aux_clean[:,i]),color='orangered',bins=50)
            else:
                plt.hist(np.log10(auxiliary_data[:,i]/jupiter_mass),color='b',bins=40)
        else:
            # plt.hist(aux_clean[:,i],color='orangered',bins=60)
            if i==1:
                plt.hist(auxiliary_data[:,i]/solar_mass,color='b',bins=40)
            elif i==2:
                plt.hist(auxiliary_data[:,i]/solar_radius,color='b',bins=40)
            elif i==7:
                plt.hist(auxiliary_data[:,i]/jupiter_radius,color='b',bins=40)
            else:
                plt.hist(auxiliary_data[:,i],color='b',bins=40)
        if i%3!=0:
            plt.yticks([])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0,3.5e4])
        plt.title(auxiliary_keys[i],fontsize=25)
    # plt.tight_layout()


def plot_target_distributions(fm_data, fm_keys):
    fig, axes = plt.subplots(1,6,figsize = (14,3))
    colors = ['darkblue','blue','royalblue','cornflowerblue','lightskyblue','lightblue']
    for i in range(len(fm_keys)):
        plt.subplot(161+i)
        plt.hist(fm_data[:,i],bins=40,color='b')
        # plt.hist(fm_clean[:,i],bins=50,color='orangered',alpha = 1)
        plt.title(fm_keys[i],fontsize=25)
        if i!=0:
        # plt.xticks([])
            plt.yticks([])
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim([0,1.2e4])
    fig.tight_layout()

def plot_feature_height_vs_mean_transitdepth(spectra):
    feature_height = np.max(spectra,axis=1)-np.min(spectra,axis=1)
    fig = plt.figure(figsize=(6,5))
    plt.scatter(np.mean(spectra,axis=1),feature_height,c=np.log10((feature_height)/7),s=0.3,cmap='tab20c')
    plt.hlines(7*1e-5,xmin=1e-5,xmax=1e-1,linestyle='-.',color='black')
    plt.hlines(7*2e-5,xmin=1e-5,xmax=1e-1,linestyle='dashed',color='black')
    plt.hlines(7*3e-5,xmin=1e-5,xmax=1e-1,linestyle='dotted',color='black')
    plt.hlines(7*5e-5,xmin=1e-5,xmax=1e-1,linestyle='solid',color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Mean Transit Depth')
    plt.ylabel('Feature Height')
    plt.title('SNR = 7')
    plt.colorbar()
    plt.tight_layout()



def plot_cov_corr_spectra_targets(x,
                                  y,
                                  y_keys,
                                  title='',
                                  correlation = False, 
                                  normalize_matrix_entries = False, 
                                  log_normalize_matrix_entries = False):
    step = 1
    tickCV = [i for i in range(0,x.shape[1]+y.shape[1],step)]
    LA = [f'$M_{str(i)[0]}$'+f'$_{str(i)[1]}$' if i>9 else f'$M_{str(i)[0]}$' for i in range(1,x.shape[1]+1)]+y_keys
    tick_labelCV = [LA[i] for i in range(0,x.shape[1]+y.shape[1],step)]
    fig = plt.figure(figsize = (10,8))

    if correlation:
        matrix = np.corrcoef(x.T,y.T)
    else: 
        matrix = np.cov(x.T,y.T)

    if normalize_matrix_entries:
        plt.imshow(matrix,cmap='bwr_r',norm=matplotlib.colors.Normalize(vmin=0.,vmax=1.))
    elif log_normalize_matrix_entries:
        plt.imshow(matrix,cmap='bwr_r',norm=matplotlib.colors.LogNorm())
    else:
        plt.imshow(matrix,cmap='bwr_r',vmin=-1.,vmax=1.)

    plt.xlabel('$d_j$',fontsize =30)
    plt.ylabel('$d_i$',fontsize =30)
    plt.title(title,fontsize =30)
    plt.xticks(tickCV,labels=tick_labelCV,fontsize=7,rotation=90)
    plt.yticks(tickCV,labels=tick_labelCV,fontsize=7)
    cb = plt.colorbar()
    if correlation:
        cb.set_label(label='corr($d_i,d_j$)',size=30)
    else: 
        cb.set_label(label='cov($d_i,d_j$)',size=30)

def plot_multiple_spectra(  wavelengths,
                            spectra,
                            y_axis_label):
    
    indices = [0,1,3,8,9]
    colors = ['red','black','magenta','blue','cyan']
    linestyles  = ['solid','dotted','dashed','dashdot','solid']
    
    fig = plt.figure(figsize = (4.8,4))

    for i, indx in enumerate(indices):
        plt.plot(wavelengths,spectra[indx],linestyle=linestyles[i],color=colors[i])
        
    plt.xlabel(r'$\lambda (\mu m)$',size=20)
    plt.ylabel(y_axis_label,size=20)
    plt.xticks(size=12)
    plt.yticks(size=12)


def plot_scatter_true_predictions(y_true,y_pred,y_keys,method):
    fig = plt.subplots(1,6,figsize=(17,3),constrained_layout=True)

    for i in range(len(y_keys)):
        plt.subplot(161+i)
        if i==0:
            plt.scatter(y_true[:,i],y_pred[:,i],c=np.abs(y_pred[:,i]-y_true[:,i]),cmap='jet',s=1,vmin=0.,vmax=1500.)
            plt.ylabel(method,fontsize=40)
            plt.xlim([0,5000])
            plt.ylim([0,5000])
        else:
            plt.scatter(y_true[:,i],y_pred[:,i],c=np.abs(y_pred[:,i]-y_true[:,i]),cmap='jet',s=1,vmin=0.,vmax=3.)
            plt.xlim([-9,-3])
            plt.ylim([-9,-3])
        if i==0 or i==len(y_keys)-1:
            plt.colorbar()#label='|Pred-True|')
        if i>1:
            plt.yticks([])
        plt.xticks([])
        # plt.xlabel('True',fontsize=25)
        plt.title(y_keys[i],fontsize=40)


def plot_mare_per(y_true,y_pred,y_keys,method,title):
    mare_col_pls = mare_per_col(y_true,y_pred) # Usual mare for each target
    inds0 = np.repeat([i for i in range(y_true.shape[0])],y_true.shape[1]) # Creates Index 0 vector
    inds1=np.argsort(y_true, axis=1)[:,::-1].flatten() # Creates Index 1 vector, sorts by largest to smallest target value
    mare_col_larg_to_small_pls = mare_per_col(y_true[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1]),
                                            y_pred[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1]))  # mare for largest to smallest target

    chart_array = np.hstack([mare_col_pls.reshape(1,mare_col_pls.shape[0]),
                             mare_col_larg_to_small_pls.reshape(1,mare_col_larg_to_small_pls.shape[0])]).reshape(1,mare_col_pls.shape[0]+mare_col_larg_to_small_pls.shape[0]) # put together
    # Plot
    fig,ax = plt.subplots(1,1,figsize=(12,2))
    im = plt.imshow(chart_array,cmap='rainbow',vmin=0,vmax=1.0)
    plt.xticks(ticks = [i for i in range(chart_array.shape[1])],labels =['$T$','$H_2O$','$CO_2$','$CH_4$','$CO$','$NH_3$','MAX','->','->','->','->','MIN'],fontsize=8)
    # plt.yticks(ticks = [i for i in range(chart_array.shape[0])],labels=methods,fontsize=25)
    plt.yticks(ticks = [])
    plt.ylabel(method,fontsize=20)
    plt.vlines(x=5.5, ymin=-0.5, ymax=0.5, colors='white', linestyles='solid',lw=5)
    # plt.title('Comparing Regression Method Errors',fontsize=25)
    plt.title(title,fontsize=25)
    cbar = plt.colorbar(im)
    cbar.set_label(label='MARE',size=25)
    cbar.ax.tick_params(labelsize=12)


def plot_model_bias_with_ytrue_bins(y_true,y_pred,y_keys,y_label_string):
    true_bins, avg_truebin_deviation, std_pred_truebins = create_true_bin_and_bias_arrays(y_true,y_pred)
    fig = plt.subplots(1,6,figsize=(18,3),constrained_layout=True)

    for i in range(len(y_keys)):
        plt.subplot(161+i)
        if i==0:
            plt.errorbar(true_bins[:,i],avg_truebin_deviation[:,i],yerr=std_pred_truebins[:,i],color='blue',ecolor='red',ms=3,marker='s',markerfacecolor='black',capsize=3)
            plt.ylabel(y_label_string,fontsize=25)
            plt.xlim([0,3500])
            plt.ylim([-250,2000])
        elif i==4:
            plt.errorbar(true_bins[:,i],avg_truebin_deviation[:,i],yerr=std_pred_truebins[:,i],color='blue',ecolor='red',ms=3,marker='s',markerfacecolor='black',capsize=3)
            plt.xlim([-6,-3])
            plt.ylim([-0.2,3])
        else:
            plt.errorbar(true_bins[:,i],avg_truebin_deviation[:,i],yerr=std_pred_truebins[:,i],color='blue',ecolor='red',ms=3,marker='s',markerfacecolor='black',capsize=3)
            plt.xlim([-9,-3])
            plt.ylim([-0.2,3])
            
        if i>1:
            plt.yticks([])
        plt.xticks([])
        # plt.xlabel('True',fontsize=20)
        plt.title(y_keys[i],fontsize=40)
    return true_bins, avg_truebin_deviation, std_pred_truebins


def plot_ypred_bins_with_average_error(y_true,y_pred,y_keys,y_label_string):
    pred_bins, avg_predbin_deviation, std_pred_predbins = create_pred_bin_and_error_arrays(y_true,y_pred)
    fig = plt.subplots(1,6,figsize=(18,3),constrained_layout=True)

    for i in range(len(y_keys)):
        plt.subplot(161+i)
        if i==0:
            plt.errorbar(avg_predbin_deviation[:,i],pred_bins[:,i],xerr=std_pred_predbins[:,i],color='blue',elinewidth=0.5,ecolor='lime',ms=3,marker='s',markerfacecolor='black',capsize=2)
            plt.ylabel(y_label_string,fontsize=30)
            plt.ylim([0,3500])
            plt.xlim([-250,2000])
        elif i==4:
            plt.errorbar(avg_predbin_deviation[:,i],pred_bins[:,i],xerr=std_pred_predbins[:,i],color='blue',elinewidth=0.5,ecolor='lime',ms=3,marker='s',markerfacecolor='black',capsize=2)
            plt.ylim([-6.2,-2.6])
            plt.xlim([-0.2,3])
        else:
            plt.errorbar(avg_predbin_deviation[:,i],pred_bins[:,i],xerr=std_pred_predbins[:,i],color='blue',elinewidth=0.5,ecolor='lime',ms=3,marker='s',markerfacecolor='black',capsize=2)
            plt.ylim([-9.2,-2.6])
            plt.xlim([-0.2,3])

        # if i>1:
        #     plt.yticks([])
        plt.xticks([])
        # plt.xlabel('True',fontsize=20)
        plt.title(y_keys[i],fontsize=40)
        # plt.xlabel(r'$\overline{|y_t-y_p|}$',fontsize=20)
    return pred_bins, avg_predbin_deviation, std_pred_predbins


def plot_histogram_error(y_true,y_pred,y_keys,method):
    fig,axs = plt.subplots(1,6,figsize=(14,2.2))

    tlim = 200
    clim = 4
    ytlim = 3000
    yclim = 13000
    ytinc = 500
    ycinc = 2000

    for i in range(len(y_keys)):
        plt.subplot(161+i)
        if i==0:
            plt.hist(y_pred[:,i]-y_true[:,i],bins=np.linspace(-tlim,tlim,30),color='black',edgecolor = "black")
            plt.ylabel(method,fontsize=25)
        else:
            plt.hist(y_pred[:,i]-y_true[:,i],bins=np.linspace(-clim,clim,30),color='black',edgecolor = "black")
        # plt.xticks(fontsize = 14)
        if i>0:
            plt.xlim([-clim,clim])
            plt.ylim([0,yclim])
            plt.xticks(fontsize=10,ticks=[-4,0,4],labels=['$-4$','0','$4$'])
            plt.yticks(ticks = [i for i in range(0,yclim+1,ycinc)],fontsize=10)
        else:
            plt.xlim([-tlim,tlim])
            plt.ylim([0,ytlim])
            plt.xticks(fontsize=10)
            plt.yticks(ticks = [i for i in range(0,ytlim+1,ytinc)],fontsize=10)
        plt.title(y_keys[i], fontsize = 25)
    fig.tight_layout()


def plot_histogram_error_maxmin(y_true,y_pred,method):
    inds0 = np.repeat([i for i in range(y_true.shape[0])],y_true.shape[1]) # Creates Index 0 vector
    inds1=np.argsort(y_true, axis=1)[:,::-1].flatten() # Creates Index 1 vector, sorts by largest to smallest target value
    keys_max_min = ['MAX','','','','','MIN']

    fig,axs = plt.subplots(1,6,figsize=(14,2.2))

    tlim = 200
    clim = 4
    ytlim = 3000
    yclim = 13000
    ytinc = 500
    ycinc = 2000

    for i in range(y_true.shape[1]):
        plt.subplot(161+i)
        if i==0:
            plt.hist(y_pred[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i]-y_true[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i],bins=np.linspace(-tlim,tlim,30),color='black',edgecolor = "black",label = keys_max_min[i])
            plt.legend(fontsize=12)
            plt.ylabel(method,fontsize=25)
        elif i==5:
            plt.hist(y_pred[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i]-y_true[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i],bins=np.linspace(-clim,clim,30),color='black',edgecolor = "black",label = keys_max_min[i])
            plt.legend(fontsize=12)
        else:
            plt.hist(y_pred[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i]-y_true[inds0,inds1].reshape(y_true.shape[0],y_true.shape[1])[:,i],bins=np.linspace(-clim,clim,30),color='black',edgecolor = "black")
        # plt.xticks(fontsize = 14)
        if i>0:
            plt.xlim([-clim,clim])
            plt.ylim([0,yclim])
            plt.xticks(fontsize=10,ticks=[-4,0,4],labels=['$-4$','0','$4$'])
            plt.yticks(ticks = [i for i in range(0,yclim+1,ycinc)],fontsize=10)
        else:
            plt.xlim([-tlim,tlim])
            plt.ylim([0,ytlim])
            plt.xticks(fontsize=10)
            plt.yticks(ticks = [i for i in range(0,ytlim+1,ytinc)],fontsize=10)
    fig.tight_layout()
 

def plot_bootstrap(y_true,
                   trials_results,
                   y_keys):
    
    M=trials_results.shape[0]
    trials = np.zeros((M+1,y_true.shape[1]+2),dtype=np.float64)
    trials[:M,:6] = trials_results
    trials[M,:6] = y_true #*targets_train_stds+targets_train_means
    trials[M,6] = 1.
    trials[:M,7] = np.ones((M,))*5.
    trials[M,7] = 20.
    trials_df = pd.DataFrame(trials,columns=y_keys+['type']+['weight']).replace([0.,1.],['pred','true'])
    sns_plot = sns.pairplot(trials_df[trials_df.keys()[:7]], 
                corner = True,
                hue='type',
                #  size = knn_trials_df['weight'],
                markers=["s", "D"],
                palette='bright',
                diag_kind = 'hist',
                plot_kws={"s":trials_df['weight']} )
                # plot_kws=dict(marker=".", s=30,c='b'),
                # diag_kws=dict(fill=True,color='b'))

    # sns_plot = sns.PairGrid(knn_trials_df[knn_trials_df.keys()[:7]], 
    #                         hue="type", 
    #                         hue_order = ['pred', 'true'], 
    #                         corner=True, 
    #                         palette='bright', 
    #                         hue_kws= {'marker':["s","D"]} )
    # sns_plot.map_diag(sns.histplot)
    # sns_plot.map_lower(sns.scatterplot, size = knn_trials_df['weight'])
    # sns_plot.add_legend(title="", adjust_subtitles=True)
    # sns_plot.tick_params(axis='both', labelsize = 10)

    for i in range(0,trials.shape[1]-2,1):
        for j in range(0,trials.shape[1]-2,1):
            if i>=j and j==0:
                sns_plot.axes[i,j].set_xlim((0,5000))
                sns_plot.axes[i,j].set_ylim((-9,-3))
                sns_plot.axes[i,j].set_xlabel(y_keys[j],fontsize=25)
                sns_plot.axes[i,j].set_ylabel(y_keys[i],fontsize=25)
            elif i>=j and j!=0:
                sns_plot.axes[i,j].set_xlim((-9,-3))
                sns_plot.axes[i,j].set_ylim((-9,-3))
                sns_plot.axes[i,j].set_xlabel(y_keys[j],fontsize=25)
                sns_plot.axes[i,j].set_ylabel(y_keys[i],fontsize=25)
    # plt.legend(fontsize=15, title_fontsize=15)


def plot_corner_differences(y_true,
                            methods_results,
                            methods,
                            y_keys):
    
    all_differences_list = []
    for i,results in enumerate(methods_results):
        all_differences_list.append(np.hstack([np.abs(results-y_true).reshape(y_true.shape[0],y_true.shape[1]), 
                                               np.ones((y_true.shape[0],1),dtype='float64')*i ]).reshape(y_true.shape[0],y_true.shape[1]+1) )

    all_differences = np.vstack([all_differences_list]).reshape(y_true.shape[0]*len(methods_results),y_true.shape[1]+1)
    
    values = [i/1. for i in range(len(methods_results))]

    differences_df = pd.DataFrame(all_differences,columns=y_keys+['Method']).replace(values,methods)
    sns_plot = sns.pairplot(differences_df, 
                corner = True,
                hue='Method',
                # markers=["s", "D"],
                palette='bright',
                diag_kind = 'hist', 
                plot_kws={"s":1.})
                # diag_kws=dict(fill=True,color='b'))

    for i in range(0,all_differences.shape[1]-1,1):
        for j in range(0,all_differences.shape[1]-1,1):
            if i>=j and j==0:
                sns_plot.axes[i,j].set_xlim((0,500))
                sns_plot.axes[i,j].set_ylim((0,3))
                sns_plot.axes[i,j].set_xlabel(y_keys[j],fontsize=25)
                sns_plot.axes[i,j].set_ylabel(y_keys[i],fontsize=25)
            elif i>=j and j!=0:
                sns_plot.axes[i,j].set_xlim((0,3))
                sns_plot.axes[i,j].set_ylim((0,3))
                sns_plot.axes[i,j].set_xlabel(y_keys[j],fontsize=25)
                sns_plot.axes[i,j].set_ylabel(y_keys[i],fontsize=25)



def plot_corr_results(y_true,
                          y_pred_method1,
                          method1,
                          y_keys,
                          y_pred_method2 = None,
                          method2 = None,
                          matrix_type = 'values'):
    
    if y_pred_method2 is not None:
        true_bins1, avg_truebin_deviation1, std_pred_truebins1 = create_true_bin_and_bias_arrays(y_true,y_pred_method1)
        true_bins2, avg_truebin_deviation2, std_pred_truebins2 = create_true_bin_and_bias_arrays(y_true,y_pred_method2)
        var_pred_truebins1 = std_pred_truebins1**2
        var_pred_truebins2 = std_pred_truebins1**2
    # pred_bins, avg_predbin_deviation, std_pred_predbins = create_pred_bin_and_error_arrays(y_true,y_pred)
    if y_pred_method2 is not None:
        if matrix_type=='bias':
            corrXY = np.corrcoef(avg_truebin_deviation1.T,avg_truebin_deviation2.T)[y_true.shape[1]:,:y_true.shape[1]]
        elif matrix_type=='variance':
            corrXY = np.corrcoef(var_pred_truebins1.T,var_pred_truebins2.T)[y_true.shape[1]:,:y_true.shape[1]]
    else:
        corrXY = np.corrcoef(y_pred_method1.T,y_true.T)[y_true.shape[1]:,:y_true.shape[1]]

    step = 1
    tickCV = [i for i in range(0,y_true.shape[1],step)]
    fig = plt.figure(figsize = (10,8))
    plt.imshow(corrXY,cmap='bwr_r',vmin=-1.,vmax=1.)
    if y_pred_method2 is not None:
        if matrix_type=='bias':
            plt.ylabel(method1+'$, \overline{|y_t-y_p|}$',fontsize =30)
            plt.xlabel(method2+'$, \overline{|y_t-y_p|}$',fontsize =30)
        elif matrix_type=='variance':
            plt.ylabel(method1+'$, \sigma^2(\overline{|y_t-y_p|})$',fontsize =30)
            plt.xlabel(method2+'$, \sigma^2(\overline{|y_t-y_p|})$',fontsize =30)
    else:
        plt.xlabel('$y_t$',fontsize =30)
        plt.ylabel(method1,fontsize =30)

    # plt.title('$\mathcal{}$',fontsize =30)
    cb = plt.colorbar()
    # cb.set_label(label='corr($d_i,d_j$)',size=30)
    plt.xticks(tickCV,labels=y_keys,fontsize=20,rotation=90)
    plt.yticks(tickCV,labels=y_keys,fontsize=20)