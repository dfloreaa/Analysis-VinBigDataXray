import os
import pandas as pd
import numpy as np
from ipywidgets import interact, interactive, IntSlider, ToggleButtons
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import random

class Datos:
    def __init__(self):
        # Crear dataframes de train y test
        self.train = pd.read_csv("train.csv")
        self.test = pd.read_csv("test.csv")
        self.classes = dict()
        self.create_classes()
        self.patologias = dict()
        self.cargar_patologias()
        

    def create_classes(self):
        # Iterar sobre todas las clases posibles, almacenadas de la forma [num],[nombre]
        with open("classes.txt") as class_list:
            for data_raw in class_list.readlines():
                data = data_raw.rstrip().split(",")
                self.classes[data[0]] = data[1]

    def n_samples(self, class_id_raw, dataframe):
        # Número de samples de cada clase dado su id, dependiendo de la dataframe (train/test)
        class_id = int(class_id_raw)
        if dataframe in ["Train", "train"]:
            return self.train.loc[self.train["class_id"] == class_id].shape[0]

        elif dataframe in ["Test", "test"]:
            return self.test.loc[self.test["class_id"] == class_id].shape[0]
            
        else:
            raise ValueError("Especificar tipo de dataframe [train] o [test]")

    def df_summary(self, dataframe = "train"):
        # Dataframe del número de samples de cada clase en la dataframe
        print(f"{dataframe.upper()} SET")

        # Total del número de samples
        total = self.train.shape[0] if dataframe in ["train", "Train"] else self.test.shape[0]
        print(f"TOTAL: {total} samples\n")

        # Cálculo de las métricas mostradas en la dataframe
        ids = list(self.classes.keys())
        clases = list(self.classes.values())
        n_samples = list(map(lambda x: self.n_samples(x, dataframe), ids))
        porcentaje = [str(round((samples/total) * 100, 1)) + "%" for samples in n_samples]

        summary = pd.DataFrame(list(zip(clases, n_samples, porcentaje)))
        summary.columns = ["Nombre de la Clase", "N° de Samples", "% del Total"]
        return summary

        # PARA HACERLO DIRECTAMENTE EN PYTHON
        # print("ID | CLASE                | NÚMERO DE SAMPLES | % DEL TOTAL ")
        # print("-----------------------------------------------------------")
        # for clase in self.classes.keys():
        #     nombre = self.classes[clase]
        #     n_ejemplos = self.n_samples(clase, dataframe)
        #     print(f"{clase:2s} - {nombre:20s} - {n_ejemplos:^17} - {(n_ejemplos/total) * 100: 8.1f}%")
        # print(f"TOTAL: {total} samples")

    def plot_summary(self, dataframe = "train"):
        # Plot del número de samples de cada clase en la dataframe

        # Cálculo de las métricas mostradas en el plot
        ids = list(self.classes.keys())
        n_samples = list(map(lambda x: self.n_samples(x, dataframe), ids))
        
        plt.xticks(rotation=90)
        return plt.bar(x = self.classes.values(), height= n_samples)

    def filter_id(self, class_id_raw, dataframe):
        # Retorna una dataframe que solo contiene tal id
        class_id = int(class_id_raw)
        if dataframe in ["train", "Train"]:
            return self.train.loc[self.train["class_id"] == class_id]
        
        elif dataframe in ["Test", "test"]:
            return self.test.loc[self.test["class_id"] == class_id]

        else:
            raise ValueError("Especificar tipo de dataframe [train] o [test]")

    def cargar_patologias(self):
        print("Cargando ejemplos de patologías...")
        for id_patologia in self.classes.keys():
            self.patologias[id_patologia] = ClasePatologia(id_patologia)
            print(".", end = "")

    def ejemplo(self, id_clase_raw, num_ejemplo = 0):
        id_clase = str(id_clase_raw)
        plt.figure(figsize = (7,7))

        pick = self.patologias[id_clase].dicom[num_ejemplo]
        nombre_clase = self.classes[id_clase]
        nombre_file = pick[1]

        plt.title(f"{nombre_clase} - {nombre_file}")
        plt.imshow(pick[0], "gray")

        # Plotear caja extraido de https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
        print(nombre_file)
        x_min = self.train.loc[self.train["image_id"] == nombre_file].x_min.iat[-1]
        y_min = self.train.loc[self.train["image_id"] == nombre_file].y_min.iat[-1]
        x_max = self.train.loc[self.train["image_id"] == nombre_file].x_max.iat[-1]
        y_max = self.train.loc[self.train["image_id"] == nombre_file].y_max.iat[-1]

        plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth = 1, edgecolor = 'r', facecolor = 'none'))

    def distribucion_pixeles(self, id_clase_raw, num_ejemplo = 0):
        id_clase = str(id_clase_raw)
        pick = self.patologias[id_clase].dicom[num_ejemplo]

        nombre_file = pick[1]
        print(f"Promedio de pixeles {np.mean(pick[0]):.4f} y Desviación Estándar {np.std(pick[0]):.4f}")

        plt.figure(figsize = (8,8))

        plt.subplot(3, 3, 1)
        plt.imshow(pick[0], "gray")
        plt.title(nombre_file)

        plt.subplot(3, 3, 3)
        sns.distplot(pick[0].ravel(), kde = False)
        plt.title(f"Distribución de Pixeles de la Imagen")
        plt.xlabel('Intensidad del Pixel')
        plt.ylabel('# de pixeles')


class ClasePatologia:
    def __init__(self, id_pat):
        self.id = id_pat
        self.dicom_raw = self.importar_dicom(fix_monochrome = False)
        self.dicom = self.importar_dicom()

    def importar_dicom(self, fix_monochrome = True):
        lista_datos = []
        aidi = str(self.id)
        path_ejemplos = os.path.join("Ejemplos", aidi)
        for imagen in os.listdir(path_ejemplos):
            if imagen.endswith(".dicom"):
                # Extraido de https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
                dicom = pydicom.read_file(os.path.join(path_ejemplos, imagen))
                data = apply_voi_lut(dicom.pixel_array, dicom)
                if dicom.PhotometricInterpretation == "MONOCHROME1" and fix_monochrome:
                    data = np.amax(data) - data
                
                data = data - np.min(data)
                data = data / np.max(data)
                data = (data * 255).astype(np.uint8)

                imagen_no_end = imagen[:-6]

                lista_datos.append((data, imagen_no_end))
                break
        return lista_datos


# a = Datos()
# print(a.ejemplo(2))
#print(pd.read_csv("train.csv").head(20))