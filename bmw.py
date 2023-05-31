# new_app1.py
import streamlit as st
import io
import pandas as pd
from streamlit.logger import update_formatter
import utils
import plotly.graph_objects as go
import numpy as np
from scipy.fft import fft, fftfreq
import plotly.express as px
import matplotlib.pyplot as plt
import os
from pathlib import Path
import math
import matplotlib.mlab as window
import shutil

st.set_option('deprecation.showfileUploaderEncoding', False)

def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n

def compute_fft(col_name,t_start,t_end):
    Fs = 4000
    T = 1/Fs 
    sample_time = t_end-t_start
    N = int(sample_time*Fs)    
    xf = fftfreq(N, T)[:N//2]
    yf = fft(df[col_name][int(t_start*Fs):int(t_end*Fs)].values)
    yf = 2.0/N * np.abs(yf[0:N//2])   
    return xf,yf

def spectro(col_name,cb_lower_limit, cb_upper_limit):

    fig,ax = plt.subplots()
    Nx = len(df[col_name])
    nsc = math.floor(Nx/400)
    nov = math.floor(nsc*0.5)
    nff = max(256,nextpow2(nsc))
    _,_,_,im = plt.specgram(df[col_name],Fs=4000.0,NFFT=nff,noverlap=nov,window= window.window_none)
    ax.set_yscale('log')
    ax.set_ylim([1.0,2e3])
    fig.colorbar(im,label='Power/Frequency (dB/Hz)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(col_name)
    plt.clim(cb_lower_limit,cb_upper_limit)
    return fig

with open('favicon.png', 'rb') as f:
    favicon = io.BytesIO(f.read())

st.set_page_config(page_title='BMW Data Exploration',
                   page_icon=favicon, 
                   layout='wide', 
                   initial_sidebar_state='expanded')
ff1,ff2 = st.beta_columns([1,6])
with ff1:
    st.image("Hutchinson.png", use_column_width=True)
st.title('BMW Data Exploration')

cc1,cc2 = st.beta_columns([1,4])
recordings_choice = ['',
 '14June21',
 '22June21',
 '13July21_2pm',
 '13July21_4pm',
 '13July21_5pm_0',
 '13July21_5pm_1']
with cc1:
    name_of_file = st.selectbox('Please select the file you want to analyze',recordings_choice)

if name_of_file:
    df = utils.read_file(name_of_file)
    

    
    time_plots = st.beta_expander('Time plots',expanded=False) 

    with time_plots:
        dd1,dd2,dd3,dd4,dd5,dd6,dd7 = st.beta_columns(7)
        dd_list = [dd2,dd3,dd4,dd5,dd6,dd7]
        with dd1:
            no_of_sensors_time = st.selectbox('Select no. of sensors to plot',['',1,2,3,4],key='time')

        if no_of_sensors_time:
            # check if all sensors are selected
            sensor_dict = {}                
            for i in range(no_of_sensors_time):
                with dd_list[i]:
                    a = st.selectbox('Select sensor-{} to plot'.format(i+1),['']+list(np.sort(df.columns[2:])),key=i+10)
                    if len(a)>1:
                        sensor_dict[i] = a
            with dd_list[i+1]:               
                t_start =  st.number_input('Insert start time (seconds)',min_value=0.0,max_value=float(df.iloc[-1,1]),value=0.0,key='t_start_time')

            with dd_list[i+2]:               
                t_end =  st.number_input('Insert end time (seconds)',min_value=t_start,max_value=float(df.iloc[-1,1]),value=60.0,key='t_end_time')
	

            # t_start, t_end = math.floor(t_start),math.floor(t_end)
            time_index = (df['Time']>= t_start) &  (df['Time']<= t_end)
            
            if len(sensor_dict.keys()) == no_of_sensors_time:
                if st.button('Plot',key='time_plot'):

                    tmpdirname_plots = 'plots'
                    tmpdirname_zip = 'archive'
                    try:
                        shutil.rmtree(tmpdirname_plots)
                        shutil.rmtree(tmpdirname_zip)
                    except:
                        pass    

                    os.makedirs(tmpdirname_plots,exist_ok=True)     
                    os.makedirs(tmpdirname_zip,exist_ok=True) 

                    fig = go.Figure()
                    for col in sensor_dict.values():
                         fig.add_trace(go.Scatter(x=df.loc[time_index]['Time'],y=df.loc[time_index][col],name=col,opacity=0.8,line=dict(width=1))) # mode='markers',marker=dict(size=1)))#
                    fig.update_layout(
        xaxis_title="Time (seconds)",
        yaxis_title="Acceleration (g)",font=dict(
                    size=18))

                    st.plotly_chart (fig, use_container_width=True) 

                    fig.write_html(tmpdirname_plots + '/' + name_of_file + '_time_plot.html')
                    utils.create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file + '_time')

        
    frequency_plots = st.beta_expander('Frequency plots',expanded=False)

    with frequency_plots:
        dd1,dd2,dd3,dd4,dd5,dd6,dd7 = st.beta_columns(7)
        dd_list = [dd2,dd3,dd4,dd5,dd6,dd7]
        with dd1:
            no_of_sensors_freq = st.selectbox('Select no. of sensors to plot',['',1,2,3,4],key='freq')

        if no_of_sensors_freq:
            # check if all sensors are selected
            sensor_dict = {}                
            for i in range(no_of_sensors_freq):
                with dd_list[i]:
                    a = st.selectbox('Select sensor-{} to plot'.format(i+1),['']+list(np.sort(df.columns[2:])),key=i+100)
                    if len(a)>1:
                        sensor_dict[i] = a
            
            # t_start = 0
            with dd_list[i+1]:
               
                t_start =  st.number_input('Insert start time (seconds)',min_value=0.0,max_value=float(df.iloc[-1,1]),value=0.0,key='t_start_freq')

            with dd_list[i+2]:
               
                t_end =  st.number_input('Insert end time (seconds)',min_value=t_start,max_value=float(df.iloc[-1,1]),value=float(df.iloc[-1,1]),key='t_end_freq')

           
            if len(sensor_dict.keys()) == no_of_sensors_freq:
                if st.button('Plot',key='freq'):

                    tmpdirname_plots = 'plots'
                    tmpdirname_zip = 'archive'
                    try:
                        shutil.rmtree(tmpdirname_plots)
                        shutil.rmtree(tmpdirname_zip)
                    except:
                        pass    

                    os.makedirs(tmpdirname_plots,exist_ok=True)     
                    os.makedirs(tmpdirname_zip,exist_ok=True) 

                    fig = go.Figure()
                    for col in sensor_dict.values():
                        xf,yf = compute_fft(col,t_start,t_end)
                        fig.add_trace(go.Scatter(x=xf,y=yf,name=col,opacity=0.8))
                        fig.update_xaxes(type="log",title_text="Frequency (Hz)")
                        fig.update_yaxes(type="log",title_text="Acceleration (g)")
                        fig.update_layout(font=dict(
                    size=18))
                    st.plotly_chart (fig, use_container_width=True)

                    fig.write_html(tmpdirname_plots + '/' + name_of_file + '_frequency_plot.html')
                    utils.create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file + '_frequency')
                    

    spectrogram_plots = st.beta_expander('Spectrogram plots',expanded=False)    

    with spectrogram_plots:
        dd1,dd2,dd3,dd4,dd5,dd6,dd7 = st.beta_columns(7)
        cb_col1,cb_col2,dummy = st.beta_columns([1,1,5])
        plot_holder = st.empty()
        dd_list = [dd2,dd3,dd4,dd5,dd6]        
        sp1,sp2,sp3,sp4 = st.beta_columns(4)

        with dd1:
            no_of_sensors_sp = st.selectbox('Select no. of sensors to plot',['',1,2,3,4],key='spectrogram')

        if no_of_sensors_sp:
            
            # check if all sensors are selected
            sensor_dict = {}                
            for i in range(no_of_sensors_sp):
                with dd_list[i]:
                    a = st.selectbox('Select sensor-{} to plot'.format(i+1),['']+list(np.sort(df.columns[2:])),key=i-101)
                    if len(a)>1:
                        sensor_dict[i] = a 

            
            with cb_col1:
                cb_lower_limit = st.number_input('Insert lower limit of color-bar (in dB)',min_value=-400.0,max_value=100.0,value=-100.0,key='col_lower')
            
            with cb_col2:
                cb_upper_limit = st.number_input('Insert upper limit of color-bar (in dB)',min_value=-400.0,max_value=100.0,value=-20.0,key='col_upper')

            sp_list = [sp1,sp2,sp3,sp4]
            if len(sensor_dict.keys()) == no_of_sensors_sp:                              
                if plot_holder.button('Plot',key='plot_spectrogram'):
                    i = 0
                    tmpdirname_plots = 'plots'
                    tmpdirname_zip = 'archive'
                    try:
                        shutil.rmtree(tmpdirname_plots)
                        shutil.rmtree(tmpdirname_zip)
                    except:
                        pass                    
                    os.makedirs(tmpdirname_plots,exist_ok=True)     
                    os.makedirs(tmpdirname_zip,exist_ok=True)                
                     
                    for col in sensor_dict.values():
                        with sp_list[i]:
                            spectrogram = spectro(col,cb_lower_limit,cb_upper_limit)
                            # create a temp folder and save all spectrograms into that folder with names as Date_sensor-name.png, zip this folder and make it available for download
                            plt.savefig(tmpdirname_plots + '/' +name_of_file+'_'+col+'.png')                            
                            st.write(spectrogram)
                            i += 1
                    
                    
                    utils.create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file + '_spectrogram')

    time_data_download = st.beta_expander('Download Data in a given interval',expanded=False)    

    with time_data_download:

                # if st.button('Download data in the interval',key='download data in interval'):
        time_1,time_2,name_of_file_col,dummy = st.beta_columns([1,1,1,4])

        with time_1:
            time_download_start = st.number_input('Insert start time (seconds)',min_value=0.0,max_value=float(df.iloc[-1,1]),value=0.0,key='t_start_download')

        with time_2:
            time_download_end = st.number_input('Insert end time (seconds)',min_value=time_download_start,max_value=float(df.iloc[-1,1]),value=float(df.iloc[-1,1]),key='t_end_download')

        name_of_file = []
        with name_of_file_col:
            name_of_file = st.text_input('Please enter file name',value='',key='file_name')

        if st.button('Download',key='download'):

            if name_of_file:
                name_of_file = name_of_file.replace(" ","_")
                tmpdirname_plots = 'plots'
                tmpdirname_zip = 'archive'
                try:
                    shutil.rmtree(tmpdirname_plots)
                    shutil.rmtree(tmpdirname_zip)
                except:
                    pass                    
                os.makedirs(tmpdirname_plots,exist_ok=True)     
                os.makedirs(tmpdirname_zip,exist_ok=True)  

                time_index = (df['Time']>= time_download_start) &  (df['Time']<= time_download_end)
                df.loc[time_index].to_csv(tmpdirname_plots +'/' +name_of_file+'.csv', sep ='\t',index=False)
                utils.create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file)
            else:
                st.error('Please enter file name')

    correlation_plots = st.beta_expander('Correlation plots (in FFT)',expanded=False)

    with correlation_plots:
        st.info('This returns- a) averaged correlation plots over chosen time interval and frequency range, b) Correlated pairs of sensors and c) Correlation between CS-side and SS-side sensors')
        dd1,dd2,dd3,dd4,dd5,dd6,dd7 = st.beta_columns(7)

        with dd1:
            t_interval =  st.number_input('Insert interval time (seconds)',min_value=1,max_value=int(df.iloc[-1,1]),value=60)
        with dd2:
            f_min =  st.number_input('Insert min. frequency',min_value=0.0,max_value=2000.0,value=0.1)
        with dd3:
            f_max =  st.number_input('Insert max. frequency',min_value=0.0,max_value=2000.0,value=500.0)
            
        plot_corr = st.button('Plot',key='plot correlation')

       
        if plot_corr:

            tmpdirname_plots = 'plots'
            tmpdirname_zip = 'archive'
            try:
                shutil.rmtree(tmpdirname_plots)
                shutil.rmtree(tmpdirname_zip)
            except:
                pass    

            os.makedirs(tmpdirname_plots,exist_ok=True)     
            os.makedirs(tmpdirname_zip,exist_ok=True) 

            t = 0
            t_end = int(df.iloc[-1,1])
            cols_of_interest = np.sort(df.columns[2:])
            count = 0
            corr_np = np.zeros((len(cols_of_interest),len(cols_of_interest)))

            while t<t_end:
                df_all_freq = pd.DataFrame()
                for col in cols_of_interest:
                    
                    xf,yf = compute_fft(col,t,t+t_interval)
                    indices = (xf >= f_min) & (xf <= f_max)
                    df_all_freq[col] = yf[indices]
                
                t += t_interval
                corr = df_all_freq.corr(method='pearson')  
                count += 1 
                corr_np += np.array(corr)

            corr_np = corr_np/count

            corr_c1,corr_c2 = st.beta_columns([1.75,1])
            with corr_c1:
                st.subheader('Correlation plot')
                fig = px.imshow(corr_np, x = cols_of_interest,y = cols_of_interest,zmin=-1.0, zmax=1.0,width=800, height=800)
                st.plotly_chart (fig, use_container_width=True)
            
            # save correlation plot
            fig.write_html(tmpdirname_plots + '/' + name_of_file + '_correlation_plot.html')

            # top-10 pairs
            list_pair = []          
            for i in corr.columns:
                for j in corr.columns:#corr_study.columns:
                    if corr.loc[i][j] < 1:
                        list_pair.append((i,j,corr.loc[i][j]))

            list_pair.sort(key = lambda x: x[2])
            list_ = list_pair[::2]
            list_.sort(key = lambda x: abs(x[2]),reverse=True)
            elem_all = []
            for elem in list_:
                first_elem = elem[0]
                second_elem = elem[1]
                
                if (elem[0].split('_')[0] != elem[1].split('_')[0]) or (elem[0].split('_')[1] != elem[1].split('_')[1]):                    
                    elem_all.append(list(elem))
            corr_result = pd.DataFrame(elem_all, columns=['Sensor-1','Sensor-2','Strength of Correlation'])

            cs_ss_list = []
            for i in corr.columns:
                for j in corr.columns:#corr_study.columns:
                    if i.split('_')[2] == 'CS' and j.split('_')[2] == 'SS' and i.split('_')[0]==j.split('_')[0] and i.split('_')[1]==j.split('_')[1] and i.split('_')[3]==j.split('_')[3]:    
                        cs_ss_list.append((i,j,corr.loc[i][j]))
            corr_resul_cs_ss = pd.DataFrame(cs_ss_list, columns=['Sensor-1-CS','Sensor-2-SS','Strength of Correlation'])

            with corr_c2:
                st.subheader('Sensors in descending order of correlation strength')
                st.write(corr_result.head(20))
                st.subheader('Chassis-side and suspension-side correlation (not in descending order)')
                st.write(corr_resul_cs_ss)
            
            corr_result.to_csv(tmpdirname_plots + '/' + name_of_file + '_correlation_strength.csv')
            corr_resul_cs_ss.to_csv(tmpdirname_plots + '/' + name_of_file + '_correlation_CS_SS_pair.csv')
            utils.create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file + '_correlation')





                   



