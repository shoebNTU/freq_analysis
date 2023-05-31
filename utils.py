import numpy as np
import pandas as pd
import streamlit as st 
from scipy.spatial.transform import Rotation as R
import shutil
import os
import base64

@st.cache(suppress_st_warning=True, show_spinner=False, allow_output_mutation=True)
def read_file(name_of_file):

# ENG_L_CS_rotM = rotx(-15.18); % 100% OK 
# ENG_R_CS_rotM = rotx(+15.28); % 100% OK 
# ENG_RR_CS_rotM = rotz(-90); % 100% OK
# FS_L_CS_rotM = rotz(0); % 100% OK  --> corrected
# FS_R_CS_rotM = rotz(180); % 100% OK
# FS_L_SS_rotM = rotx(16.91)*rotx(-90)*rotz(90); % 100% OK 
# FS_R_SS_rotM = rotx(-20.3)*rotx(-90)*rotz(90); % 100% OK 
# RS_L_CS_rotM = rotz(0); % 100% OK
# RS_R_CS_rotM = rotz(180); % 100% OK
# RS_L_SS_rotM = [0 -1 0 ; -1 0 0 ; 0 0 -1]; % 100% OK
# RS_R_SS_rotM = rotx(9.17)*rotx(-90)*rotz(90); % 100% OK 
# ENG_L_SS_rotM = roty(12.43)*rotz(15.18)*roty(90); % 100% OK 
# ENG_R_SS_rotM = roty(28.7)*rotx(-11.42)*rotz(-90); % 100% OK 
# ENG_RR_SS_rotM = rotz(0); % 100% OK

    rot_dict = {'ENG_L_CS':R.from_euler('x', -15.18, degrees=True).as_matrix(),
           'ENG_R_CS':R.from_euler('x', 15.28, degrees=True).as_matrix(),
            'ENG_RR_CS':R.from_euler('z', -90, degrees=True).as_matrix(),
            'FS_L_CS':R.from_euler('z', 0, degrees=True).as_matrix(),
            'FS_R_CS':R.from_euler('z', 180, degrees=True).as_matrix(),
            'FS_L_SS':R.from_euler('zx', [90,-90+16.91], degrees=True).as_matrix(),
            'FS_R_SS':R.from_euler('zx', [90,-90-20.3], degrees=True).as_matrix(),
            'RS_L_CS':R.from_euler('z', 0, degrees=True).as_matrix(),
            'RS_R_CS':R.from_euler('z', 180, degrees=True).as_matrix(),
            'RS_L_SS':np.array([[0,-1,0],[-1, 0, 0],[0,0,-1]]),
            'RS_R_SS':R.from_euler('zx', [90,-90+9.17], degrees=True).as_matrix(),
            'ENG_L_SS':R.from_euler('yzy', [90,15.18,12.43], degrees=True).as_matrix(),
            'ENG_R_SS':R.from_euler('zxy', [-90,-11.42,28.7], degrees=True).as_matrix(),
            'ENG_RR_SS':R.from_euler('z', 0, degrees=True).as_matrix()            
           }
           # FS_L_CS_X/Z and ENG_RR_SS_X/Z --> suspicious ones, possibly swapped
    df = pd.read_csv('../Data/'+name_of_file+'.csv',sep='\t')
    df.dropna(axis=1,how='any',inplace=True) 

    temp = df['FS_L_CS_X']
    df['FS_L_CS_X'] = df['FS_L_CS_Z']
    df['FS_L_CS_Z'] = temp

    temp = df['ENG_RR_SS_X']
    df['ENG_RR_SS_X'] = df['ENG_RR_SS_Z']
    df['ENG_RR_SS_Z'] = temp

    # for sensor_name in rot_dict.keys():
    #     col_to_accum = []
    #     for col in np.sort(df_all_new.columns[2:]):    
    #         if sensor_name in col:
    #             col_to_accum.append(col)

    #     df_all_new[col_to_accum] = (np.dot(rot_dict[sensor_name],(np.array(df_all_new[col_to_accum])).T)).T

    for sensor_name in rot_dict.keys():
        # if sensor_name in list(df.columns):
        col_to_accum = []
        for col in np.sort(df.columns[2:]):    
            if sensor_name in col:
                col_to_accum.append(col)
        
        if col_to_accum:
            df[col_to_accum] = (np.dot(rot_dict[sensor_name],(np.array(df[col_to_accum])).T)).T  

    for col in df.columns[2:]:
        col_mean = np.nanmean(df[col])
        if 'CS' in col:
            scaling_factor = 0.3
        else:
            if 'ENG' in col:
                scaling_factor = 0.02
            else:
                scaling_factor = 0.057
        
        df[col] = (df[col]-col_mean)/scaling_factor
    
    # tmpdirname_plots = 'csv_file'
    # tmpdirname_zip = 'archive_csv'
    # try:
    #     shutil.rmtree(tmpdirname_plots)
    #     shutil.rmtree(tmpdirname_zip)
    # except:
    #     pass                    
    # os.makedirs(tmpdirname_plots,exist_ok=True)     
    # os.makedirs(tmpdirname_zip,exist_ok=True) 

    # df.to_csv(tmpdirname_plots + '/' + name_of_file + '.csv')
    # create_download_zip(tmpdirname_plots,tmpdirname_zip,name_of_file + '_csvfile')

    return df

def create_download_zip(zip_directory, zip_path, filename="foo.zip"):
    """ 
        zip_directory (str): path to directory  you want to zip 
        zip_path (str): where you want to save zip file
        filename (str): download filename for user who download this
    """
    shutil.make_archive(os.path.join(zip_path,filename), 'zip', zip_directory)
    with open(os.path.join(zip_path,filename+'.zip'), 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/zip;base64,{b64}" download=\'{filename}.zip\'>\
            Click here to download \
        </a>'
        st.markdown(href, unsafe_allow_html=True)
