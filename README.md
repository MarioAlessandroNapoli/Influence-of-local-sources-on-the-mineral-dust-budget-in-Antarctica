# Tesi
Needed data can be found here:
https://drive.google.com/file/d/1IQ83KYdd2uWjKcCUo3yhWjWW4ovhJiu0/view?usp=sharing

to execute the code of the first step put theese files inside: data/step_1_data_input

To get started:
'''
git clone https://github.com/MarioAlessandroNapoli/Tesi.git
cd Tesi
conda create --name tesi_env
conda activate tesi_env
conda install -c conda-forge gdal
conda install geopandas
conda install -c conda-forge rasterio
pip install -r requirements.txt
python -m ipykernel install --user --name tesi_env
'''
Done that 'tesi_env' will be a selectable kernel inside Jupyter Notebook or Lab and all the dependecies will be installed.
