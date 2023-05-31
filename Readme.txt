Dependencies

An installation of python:
conda (recommended)
anaconda: https://www.anaconda.com/products/individual
miniconda: https://docs.conda.io/en/latest/miniconda.html
activate option to add environment variables to path during installation
python:
version 3.7.6: https://www.python.org/downloads/

Setup
Install conda environment:
with command prompt navigate to project folder 
conda env create -f bmw_env.yml

Streamlit modification
Go to site-packages/streamlit/server/server_util.py, and change the parameter MESSAGE_LIMIT_SIZE to 500*1e6 (or any other size as per requirement, by default the size is 50 MB - and often times the program runs into out of memory problem)

Running the code
run app:
with command prompt and navigate to project folder 
conda activate bmw_analysis
streamlit run bmw.py (if running on VM --> streamlit run bmw.py --server.port 80)

PS:
In utils.py file, please correct according to the path where you place your csv files in the below code - 
df = pd.read_csv('../Data/'+name_of_file+'.csv',sep='\t')