## M. Sc. Thesis
[Influenza delle sorgenti locali sul bilancio di polveri minerali in Antartide](Napoli_MarioAlessandro_Tesi_LMDS_09-06-2022.pdf)


### Installation 
Needed data can be found here:
https://drive.google.com/drive/folders/1BsmqEwzjdKfLA-8_PhSMPqQfkpNcMI5l?usp=sharing

To get started:
```
git clone https://github.com/MarioAlessandroNapoli/Tesi.git
cd Tesi
conda create --name tesi_env
conda activate tesi_env
conda install -c conda-forge gdal
conda install -c conda-forge cartopy
conda install --channel conda-forge geopandas
conda install -c conda-forge rasterio
conda install ipywidgets
pip install -r requirements.txt
python -m ipykernel install --user --name tesi_env
```

Done that 'tesi_env' will be a selectable kernel inside Jupyter Notebook or Lab and all the dependecies will be installed.

