import time
import urllib.request
from tqdm.auto import tqdm
from pathlib import Path
import ssl

# Crea una carpeta nueva, "Mapas", en la que guarda los archivos .gz.
basepath = Path('./Mapas de Segmentación')
path = basepath / 'Mapas'
path.mkdir(exist_ok=True)

# Bajo el archivo .sha1sum en el que están los nombres de todos los archivos .gz
# que queremos bajar y sus respectivos sha1sum.
url = 'https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/'
sha1sum = 'manga_morphology_galaxyzoo3d_v4_0_0.sha1sum'
urllib.request.urlretrieve(url+sha1sum, basepath/sha1sum)

archivos = open(basepath/sha1sum, 'r').read().split('\n')[:-1]
gzs = [a.split(' ')[2] for a in archivos]
sha1sums = [a.split(' ')[0] for a in archivos]

# Creo un contexto que no verifica el SSL certificate porque me tiraba un error.
# No es lo más óptimo pero con la página del SDSS no pasa nada.
ssl._create_default_https_context = ssl._create_unverified_context

# Descargo todos los archivos comprimidos en .gz en la carpeta "Mapas".
t_i = time.time()
for filename in tqdm(gzs):
    urllib.request.urlretrieve(url+filename, path/filename)

t_f = time.time()-t_i
print('Tardó ' + str(int(t_f/3600)) + ' horas y ' + str(round((t_f/3600 % 1)*60)) + ' minutos en descargar todos los archivos')
