import os
import random
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
    def __init__(self, all_examples=True):
        # Crear dataframes de train y test
        self.train = pd.read_csv("train.csv")
        self.test = pd.read_csv("test.csv")
        self.classes = dict()
        self.create_classes()
        self.patologias = dict()
        self.cargar_patologias(all_examples)

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
            return self.test.loc[self.test["image_id"] == class_id].shape[0]

        else:
            raise ValueError("Especificar tipo de dataframe [train] o [test]")

    def df_summary(self, dataframe="train"):
        # Dataframe del número de samples de cada clase en la dataframe
        print(f"{dataframe.upper()} SET\n")

        # Total del número de samples
        total = self.train.shape[0] if dataframe in ["train", "Train"] else self.test.shape[0]
        print(f"TOTAL DE LABELS (Número de filas): {total} labels")
        samples_unicas = self.train.image_id.unique().shape[0] if dataframe in [
            "train", "Train"] else self.test.image_id.unique().shape[0]
        print(f"TOTAL DE SAMPLES ÚNICAS: {samples_unicas} samples\n")

        # Cálculo de las métricas mostradas en la dataframe
        ids = list(self.classes.keys())
        clases = list(self.classes.values())
        n_samples = list(map(lambda x: self.n_samples(x, dataframe), ids))
        porcentaje = [str(round((samples / total) * 100, 1)) + "%" for samples in n_samples]

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

    def plot_summary(self, dataframe="train"):
        # Plot del número de samples de cada clase en la dataframe

        # Cálculo de las métricas mostradas en el plot
        ids = list(self.classes.keys())
        n_samples = list(map(lambda x: self.n_samples(x, dataframe), ids))

        plt.xticks(rotation=90)
        return plt.bar(x=self.classes.values(), height=n_samples)

    def filter_id(self, class_id_raw, dataframe):
        # Retorna una dataframe que solo contiene tal id
        class_id = int(class_id_raw)
        if dataframe in ["train", "Train"]:
            return self.train.loc[self.train["class_id"] == class_id]

        elif dataframe in ["Test", "test"]:
            return self.test.loc[self.test["class_id"] == class_id]

        else:
            raise ValueError("Especificar tipo de dataframe [train] o [test]")

    def cargar_patologias(self, load_all):
        print("Cargando ejemplos de 14 patologías distintas...")
        for id_patologia in self.classes.keys():
            self.patologias[id_patologia] = ClasePatologia(id_patologia, load_all)
            print(".", end="")

    def ejemplo(self, id_clase_patologia, num_ejemplo=0):
        id_clase = str(id_clase_patologia)
        plt.figure(figsize=(7, 7))

        pick = self.patologias[id_clase].dicom[num_ejemplo]
        nombre_clase = self.classes[id_clase]
        nombre_file = pick[1]

        plt.title(f"{nombre_clase} - {nombre_file}")
        plt.imshow(pick[0], "gray")

        # Plotear caja extraido de https://stackoverflow.com/questions/37435369/matplotlib-how-to-draw-a-rectangle-on-image
        print(nombre_file)

        for index, row in self.train.loc[self.train["image_id"] == nombre_file].iterrows():
            class_id = str(row["class_id"])

            if class_id == id_clase:
                x_min = row["x_min"]
                y_min = row["y_min"]
                x_max = row["x_max"]
                y_max = row["y_max"]

                plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min,
                                              y_max - y_min, linewidth=1, edgecolor="r", facecolor='none'))

    def distribucion_pixeles(self, id_clase_patologia, num_ejemplo=0):
        id_clase = str(id_clase_patologia)
        pick = self.patologias[id_clase].dicom[num_ejemplo]

        nombre_file = pick[1]
        print(
            f"Promedio de pixeles {np.mean(pick[0]):.4f} y Desviación Estándar {np.std(pick[0]):.4f}")

        plt.figure(figsize=(8, 8))

        plt.subplot(3, 3, 1)
        plt.imshow(pick[0], "gray")
        plt.title(nombre_file)

        plt.subplot(3, 3, 3)
        sns.distplot(pick[0].ravel(), kde=False)
        plt.title(f"Distribución de Pixeles de la Imagen")
        plt.xlabel('Intensidad del Pixel')
        plt.ylabel('# de pixeles')

    def ejemplos_random_all(self):
        plt.figure(figsize=(20, 20))
        for i in range(14):
            id_clase = str(i)
            pick = random.choice(self.patologias[id_clase].dicom)

            plt.subplot(4, 4, i + 1)
            plt.imshow(pick[0], "gray")
            plt.title(self.classes[str(i)])
            nombre_file = pick[1]
            for _, row in self.train.loc[self.train["image_id"] == nombre_file].iterrows():
                class_id = str(row["class_id"])

                if class_id == id_clase:
                    x_min = row["x_min"]
                    y_min = row["y_min"]
                    x_max = row["x_max"]
                    y_max = row["y_max"]

                    plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min,
                                                  y_max - y_min, linewidth=1, edgecolor="r", facecolor='none'))

    def all_pat_img(self, num_ejemplo=0):
        carpeta = str(int((num_ejemplo - 1) / 5))
        pick = self.patologias[carpeta].dicom[num_ejemplo - (int(num_ejemplo / 5) * 5)]
        nombre_file = pick[1]
        plt.imshow(pick[0], "gray")

        colores = {"0": "tab:blue", "1": "tab:orange", "2": "tab:green", "3": "tab:red", "4": "tab:purple", "5": "tab:brown", "6": "tab:pink",
                   "7": "tab:gray", "8": "tab:olive", "9": "tab:cyan", "10": "gold", "11": "lime", "12": "mediumpurple", "13": "azure", "14": None}

        for _, row in self.train.loc[self.train["image_id"] == nombre_file].iterrows():
            x_min = row["x_min"]
            y_min = row["y_min"]
            x_max = row["x_max"]
            y_max = row["y_max"]
            class_id = str(row["class_id"])

            plt.gca().add_patch(Rectangle((x_min, y_min), x_max - x_min, y_max -
                                          y_min, linewidth=1, edgecolor=colores[class_id], facecolor='none'))


    def df_r_summary(self, order = "No"):

        if order:
            if order not in ["asc", "dsc"]:
                raise ValueError("Por favor señala alguna opción entre asc/dsc/False")

        # Dataframe del número de samples de cada clase en la dataframe
        print(f"TRAIN SET\n")

        # Total del número de samples
        total = self.train.shape[0]
        print(f"TOTAL DE LABELS (Número de filas): {total} labels")
        samples_unicas = self.train.image_id.unique().shape[0]
        print(f"TOTAL DE SAMPLES ÚNICAS: {samples_unicas} samples\n")

        # Cálculo de labels para cada radiologo
        ids = ['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10',
       'R11', 'R12', 'R13', 'R14', 'R15', 'R16', 'R17']
        n_samples = list(map(lambda x: self.n_samples_for_r(x), ids))
        porcentaje = [str(round((samples / total) * 100, 1)) + "%" for samples in n_samples]

        summary = pd.DataFrame(list(zip(ids, n_samples, porcentaje)))
        summary.columns = ["ID de Radiolog@", "N° de Samples", "% del Total"]
        
        if order == "asc" or order == "dsc":
            summary["% del Total"] = summary["% del Total"].map(lambda x: float(x[:-1]))
            # FALSE PARA DESCENDIENTE
            orden = True if order == "asc" else False
            summary = summary.sort_values("% del Total", ascending = orden)
            summary["% del Total"] = summary["% del Total"].map(lambda x: str(x) + "%")
        
        return summary

    def n_samples_for_r(self, rad_id):
        # Número de samples de cada radiólogo dado su id
        return self.train.loc[self.train["rad_id"] == rad_id].shape[0]

class ClasePatologia:
    def __init__(self, id_pat, load_all_ex):
        self.id = id_pat
        self.dicom_raw = self.importar_dicom(fix_monochrome=False)
        self.dicom = self.importar_dicom(load_all = load_all_ex)

    def importar_dicom(self, fix_monochrome=True, load_all=True):
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
                if not load_all: break
                else: continue
        return lista_datos


# a = Datos()
# print(a.ejemplo(2))
# print(pd.read_csv("train.csv").head(20))
