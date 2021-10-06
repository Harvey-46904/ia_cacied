
import glob
import base64
import urllib
import io
from django.http import JsonResponse
from django.shortcuts import render
from django.http import response
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import os
from os import system
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout, MaxPooling2D, Dense
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
from .forms import PersonaForm, pacienteForm, CitaForm, ImageForm
from .models import *
from django.shortcuts import redirect
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import ipywidgets as widgets
from PIL import Image
from IPython.display import display, clear_output
from warnings import filterwarnings
from keras.applications.vgg19 import VGG19, preprocess_input
import glob
# vistas de la paginas


def inicios(request):
    return render(request, 'dashboard.html')


def login(request):
    return render(request, 'login.html')


def vista_profesional(request):
    return render(request, 'profesionales.html')


def lista_profesionales(request):
    profesionales = profesional.objects.all()
    contexto = {
        'profesional': profesionales
    }
    return render(request, 'mirar_profesionales.html', contexto)


def registro_profesional(request):
    if request.method == 'GET':
        form = PersonaForm()
        contexto = {
            'form': form
        }
    else:
        form = PersonaForm(request.POST)
        contexto = {
            'form': form
        }
        if form.is_valid():
            print("data")
            form.save()
            return redirect("listar_registro")
        print(form)
    return render(request, 'registro_profesional.html', contexto)


def vista_paciente(request):
    return render(request, 'paciente.html')


def lista_paciente(request):
    profesionales = Paciente.objects.all()
    contexto = {
        'profesional': profesionales
    }
    return render(request, 'mirar_paciente.html', contexto)


def registro_paciente(request):
    if request.method == 'GET':
        form = pacienteForm()
        contexto = {
            'form': form
        }
    else:
        form = pacienteForm(request.POST)
        contexto = {
            'form': form
        }
        if form.is_valid():
            print("data")
            form.save()
            return redirect("listar_paciente")
        print(form)
    return render(request, 'registro_paciente.html', contexto)


def lista_citas(request):
    profesionales = Cita.objects.all()
    contexto = {
        'profesional': profesionales
    }
    return render(request, 'mirar_cita.html', contexto)


def registro_citas(request):
    if request.method == 'GET':
        form = CitaForm()
        contexto = {
            'form': form
        }
    else:
        cita = Cita()
        cita.id_profesional = profesional.objects.get(
            id=request.POST.get("id_profesional"))
        cita.id_paciente = Paciente.objects.get(
            id=request.POST.get("id_paciente"))
        cita.resonancia = request.FILES.get("resonancia")
        cita.save()
        return redirect("listar_cita")
        print("ok")
    return render(request, 'registro_cita.html', contexto)


def logins(request):
    user = request.POST.get("email")
    passw = request.POST.get("password")
    if user == "admin@gmail.com" and passw == "admin":
        return redirect("profesion")
    else:
        profesionales = profesional.objects.filter(
            correo=user, contraseña=passw).count()

        print(profesionales)
        if profesionales == 1:
            return redirect("paciente")
    return redirect("login")


''' 
llama al archivo .h5 que se genero en el entrenamiento y este lo guarda 
para obtener parametros segun la imagen de entreda 
'''


def cargar_modelo(url):
    json_file = open('./static/modelo/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # cargar pesos al nuevo modelo
    loaded_model.load_weights("./static/modelo/model.h5")
    print("Cargado modelo desde disco.")

    # Compilar modelo cargado y listo para usar.
    loaded_model.compile(loss='categorical_crossentropy',
                         optimizer='Adam', metrics=['accuracy'])
    # imagen

    image = cv2.imread(url)
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = loaded_model.predict(img)
    p = np.argmax(p, axis=1)[0]
    if p == 0:
        p = 'Glioma Tumor'
        return p
    elif p == 1:
        p = 'No Tumor'

        print('The model predicts that there is no tumor')
        return p
    elif p == 2:
        p = 'Meningioma Tumor'
        return p
    else:
        p = 'Pituitary Tumor'
        return p
    if p != 1:
        print(f'The Model predicts that it is a {p}')
    return p


def analizar(request, id):
    persona = Paciente.objects.get(id=id)
    resonancia = Cita.objects.get(id=persona.id)
    url = "./"+resonancia.resonancia.url
    prediccion = cargar_modelo(url)
    #uri = mostrar_imagen(url)

    tumor = {
        0: "Glioma Tumor",
        1: "No Tumor",
        2: "Meningioma Tumor",
        3: "Pituitary Tumor"
    }

    contexto = {
        'profesional': persona,
        'resonancia': resonancia,
        'prediccion': prediccion
    }

    return render(request, 'cita_sola.html', contexto)


def modulos(request):
    return render(request, 'menu.html')


def imagenes(request):
    form = ImageForm(request.POST, request.FILES)

    form.save()
    return JsonResponse({'you name': "carga correcta"})


def check_folders(request):
    ''' revisa las carpetas que estan en el proyecto
    donde estan clasificados los tumores'''
    path = "./static/input/brain-tumor-classification-mri"
    labels = ['glioma_tumor', 'no_tumor',
              'meningioma_tumor', 'pituitary_tumor']
    '''le asignamos un id a cada tumor para ser detectado'''
    class_map = {
        'no_tumor': 0,
        'glioma_tumor': 1,
        'meningioma_tumor': 2,
        'pituitary_tumor': 3
    }

    inverse_class_map = {
        0: 'no_tumor',
        1: 'glioma_tumor',
        2: 'meningioma_tumor',
        3: 'pituitary_tumor'
    }

    '''parametros para configurar imagen
    se confnigura para todas las imagenes 
    el mismo ancho y alto'''
    h, w = 224, 224
    batch_size = 32
    epochs = 100

    '''Leer la imagen y guardarla como np.array, 
    junto con sus respectivas etiquetas'''
    IMAGE = []
    LABELS = []
    '''recorre cada imagen y la va clasificando con
    la etiqueta y a su vez convirtiendola en un arreglo'''
    for label in labels:
        ''' path join conbina los nombres de la ruta en una ruta completa'''
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri/Training', label)
        for j in tqdm(os.listdir(folderPath)):
            ''' imread como tal cv2 es una bibliteca para resolver problemas de vision por computadora en este caso el metodo devuelve un archivo'''
            img = cv2.imread(os.path.join(folderPath, j))
            '''rezice redimenciona una imagen se puede cambiar su hight y weith '''
            img = cv2.resize(img, (h, w))
            IMAGE.append(img)
            LABELS.append(class_map[label])

    for label in labels:
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri/Testing', label)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath, j))
            img = cv2.resize(img, (h, w))
            IMAGE.append(img)
            LABELS.append(class_map[label])

    '''al final optenemos un array tanto de imagenes
    como de etiquetas'''
    X = np.array(IMAGE)
    y = np.array(LABELS)
    print("this array image and labels")
    print(X)
    print(y)
    '''retornamos la direccion de la carpeta'''
    return JsonResponse(
        {'files': list(os.listdir(path + "/Training"))})


def pruebas(request):
    '''seleccionamos 4 imagenes de entrenamientoo
    y las transformamos en bits correspondientes
    para el entrenamiento 
    este retorona la imagen estudiada 
    como tambien puede retornar el grafico de dispercion
    '''
    plt.figure(figsize=(16, 12))

    h, w = 224, 224

    path = './static/input/brain-tumor-classification-mri/Training/'
    fileNames = ['glioma_tumor/gg (10).jpg', 'meningioma_tumor/m (108).jpg',
                 'no_tumor/image (16).jpg', 'pituitary_tumor/p (12).jpg']
    fileLabels = ['glioma_tumor', 'meningioma_tumor',
                  'no_tumor', 'pituitary_tumor']
    for i in range(4):
        ax = plt.subplot(4, 4, i + 1)
        img = mpimg.imread(path + fileNames[i])
        img = cv2.resize(img, (h, w))
        plt.imshow(img)
        plt.title(fileLabels[i])
        plt.axis("off")

    plt.plot(range(10))
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)
    return render(request, 'prueba.html', {'data': uri})


def cargar_imagenes():
    path = "./static/input/brain-tumor-classification-mri"
    labels = ['glioma_tumor', 'no_tumor',
              'meningioma_tumor', 'pituitary_tumor']
    '''le asignamos un id a cada tumor para ser detectado'''
    class_map = {
        'no_tumor': 0,
        'glioma_tumor': 1,
        'meningioma_tumor': 2,
        'pituitary_tumor': 3
    }

    inverse_class_map = {
        0: 'no_tumor',
        1: 'glioma_tumor',
        2: 'meningioma_tumor',
        3: 'pituitary_tumor'
    }

    '''parametros para configurar imagen
    se confnigura para todas las imagenes 
    el mismo ancho y alto'''
    h, w = 32, 32
    batch_size = 32
    epochs = 100

    '''Leer la imagen y guardarla como np.array, 
    junto con sus respectivas etiquetas'''
    IMAGE_TRAIN = []
    LABELS_TRAIN = []

    IMAGE_TEST = []
    LABELS_TEST = []
    '''recorre cada imagen y la va clasificando con
    la etiqueta y a su vez convirtiendola en un arreglo'''
    for label in labels:
        ''' path join conbina los nombres de la ruta en una ruta completa'''
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri/Training', label)
        for j in tqdm(os.listdir(folderPath)):
            ''' imread como tal cv2 es una bibliteca para resolver problemas de vision por computadora en este caso el metodo devuelve un archivo'''
            img = cv2.imread(os.path.join(folderPath, j))
            '''rezice redimenciona una imagen se puede cambiar su hight y weith '''
            img = cv2.resize(img, (h, w))
            IMAGE_TRAIN.append(img)
            LABELS_TRAIN.append(class_map[label])

    for label in labels:
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri/Testing', label)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath, j))
            img = cv2.resize(img, (h, w))
            IMAGE_TEST.append(img)
            LABELS_TEST.append(class_map[label])

    '''al final optenemos un array tanto de imagenes
    como de etiquetas'''
    train_imagen = np.array(IMAGE_TRAIN)
    train_label = np.array(LABELS_TRAIN)

    test_imagen = np.array(IMAGE_TEST)
    test_label = np.array(LABELS_TEST)

    return (train_imagen, train_label, test_imagen, test_label)


def categorizar(y_train, y_test):
    y_train_one_hot = to_categorical(y_train)
    x_test_one_hot = to_categorical(y_test)
    return (y_train_one_hot, x_test_one_hot)


def entrenar(train_imagen, y_train_one_hot, x_test_one_hot):
    '''modelo de arquitectura'''
    model = Sequential()
    # agregar primera capa
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    # agregar poligoonos a capa
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # agregar capa de compulucion
    model.add(Conv2D(32, (5, 5), activation='relu'))
    # agregar poligoonos a capa
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # capa de aplanamiento
    model.add(Flatten())
    # agregar capa con 500 neuronas
    model.add(Dense(500, activation='relu'))
    # agregar y arrastrar capa
    model.add(Dropout(0.5))

    # agregar capa con 250 neuronas
    model.add(Dense(250, activation='relu'))

    # agregar capa con 10 neuronas
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    system("cls")
    hist = model.fit(train_imagen, y_train_one_hot,
                     batch_size=256, epochs=10, validation_split=0.2)
    print("ok")

    '''
   
    hist = model.fit(train_imagen, y_train_one_hot,
                     batch_size=256, epochs=10, validation_split=0.2)
    model.evaluate(train_imagen, x_test_one_hot)[1]
    '''


def mensaje(request):
    index = 12
    (train_imagen, train_label, test_imagen, test_label) = cargar_imagenes()
    plt.imshow(train_imagen[index])
    plt.title("figura")
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    etiqueta = train_label[index]

    classification = ['no_tumor', 'glioma_tumor',
                      'meningioma_tumor', 'pituitary_tumor']

    clasificacion = classification[train_label[index]]

    (y_train_one_hot, x_test_one_hot) = categorizar(train_label, test_label)

    print("caliente", y_train_one_hot.shape)
    print("caliente train", train_imagen.shape)

    print("nueva etiqueta de imagen: ", y_train_one_hot[index])

    '''normalizar'''
    train_imagen = train_imagen/255
    test_imagen = test_imagen/255

    print("nuevos valores")
    print(train_imagen[index])

    #entrenar(train_imagen, y_train_one_hot, x_test_one_hot)
    return render(
        request, 'prueba.html',
        {
            'train_imagen': train_imagen.shape,
            'train_label': train_label.shape,
            'test_imagen': test_imagen.shape,
            'test_label': test_label.shape,
            'first': train_imagen[1],
            'imagen': uri,
            'etiqueta': etiqueta,
            'clasificacion': clasificacion
        })


def loading_image():
    for dirname, _, filenames in os.walk('/static/input/'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# ============ ENTRENAMIENTO IA ==============


ruta_input = "./static/input"
colors_dark = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
colors_red = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
colors_green = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']
log_carga_imagenes = []
X_train = []
y_train = []
image_size = 150
# variables de etiquetas
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']
y_train_new = []
y_test_new = []
y_test = []
X_test = []
sumary = []
informacion = ""

# cargamos las imagenes en un array


def carga_de_imagenes():
    for dirname, _, filenames in os.walk(ruta_input):
        for filename in filenames:
            log_carga_imagenes.append(os.path.join(dirname, filename))


'''
agregamos todas las imágenes de los directorios
 en una lista de Python y luego las 
convertimos en matrices numerosas después de cambiar su tamaño
a un valor de 150
'''


def convertir_imagenes():
    global X_train
    global y_train
    for i in labels:
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri', 'Training', i)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath, j))
            img = cv2.resize(img, (image_size, image_size))
            X_train.append(img)
            y_train.append(i)

    for i in labels:
        folderPath = os.path.join(
            './static/input/brain-tumor-classification-mri', 'Testing', i)
        for j in tqdm(os.listdir(folderPath)):
            img = cv2.imread(os.path.join(folderPath, j))
            img = cv2.resize(img, (image_size, image_size))
            X_train.append(img)
            y_train.append(i)

    X_train = np.array(X_train)
    y_train = np.array(y_train)


'''
una ves realizado el proceso de conversion se procede asignarlas en una matriz
con su respectiva etiqueta
'''


def muestreo_imagenes():
    k = 0
    fig, ax = plt.subplots(1, 4, figsize=(20, 20))
    fig.text(s='Sample Image From Each Label', size=18, fontweight='bold',
             fontname='monospace', color=colors_dark[1], y=0.62, x=0.4, alpha=0.8)
    for i in labels:
        j = 0
        while True:
            if y_train[j] == i:
                ax[k].imshow(X_train[j])
                ax[k].set_title(y_train[j])
                ax[k].axis('off')
                k += 1
                break
            j += 1


'''
aumentamos al azar el tamaño de datos de imagenes 
con esto obtenemos artificialmente un conjunto de datos mas grande
con sus recpectivas modificaciones en zoom,rellenar ,recortar

'''


def array_imagen():
    global X_train
    global y_train
    X_train, y_train = shuffle(X_train, y_train, random_state=101)
    X_train.shape
    print(X_train.shape)


'''
al finalizar se obtine un aumento de datos mas robustos
y sobrescribimos el entrenamiento de imagenes y labels

'''


def entrenamiento_test():
    global X_train
    global y_train
    global y_test
    global X_test
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=101)


def entrenando_labels():
    global y_train
    for i in y_train:
        y_train_new.append(labels.index(i))
    y_train = y_train_new
    y_train = tf.keras.utils.to_categorical(y_train)


'''
Dividir el conjunto de datos en conjuntos de entrenamiento y prueba.
'''


def entrenando_labels_prueba():
    global y_test
    for i in y_test:
        y_test_new.append(labels.index(i))
    y_test = y_test_new
    y_test = tf.keras.utils.to_categorical(y_test)


'''
Realización de una codificación en caliente en las etiquetas
 después de convertirla en valores numéricos:
'''


def descarga_modelo():
    ms = EfficientNetB0(weights='imagenet', include_top=False,
                        input_shape=(image_size, image_size, 3))
    return ms


'''
se prepara el escalado de imagenes por medio de efficienet 
este aplica variaciones pequeñas a las imagenes procesadas por medio de imageNet
'''


'''
GlobalAveragePooling2D -> Esta capa actúa de manera similar a la capa 
Max Pooling en las CNN, la única 
diferencia es que usa los valores promedio en lugar del
 valor máximo mientras se agrupa. Esto realmente ayuda a disminuir
  la carga computacional en la máquina durante el entrenamiento.

Abandono -> Esta capa omite algunas de las neuronas en cada paso de 
la capa, lo que hace que las neuronas sean más independientes de las 
neuronas vecinas. Ayuda a evitar el sobreajuste. Las neuronas a 
omitir se seleccionan al azar. El parámetro de frecuencia es la 
probabilidad de que la activación de una neurona se establezca
 en 0, eliminando así la neurona

Denso -> Esta es la capa de salida que clasifica la imagen en 
1 de las 4 clases posibles. Utiliza la función softmax que es
 una generalización de la función sigmoidea.

'''


def instructor_modelo():
    global model
    model = tf.keras.layers.GlobalAveragePooling2D()(model)
    model = tf.keras.layers.Dropout(rate=0.5)(model)
    model = tf.keras.layers.Dense(4, activation='softmax')(model)
    model = tf.keras.models.Model(inputs=effnet.input, outputs=model)
    model.summary(print_fn=lambda x: sumary.append(x))
    short_model_summary = "\n".join(sumary)
    return short_model_summary


'''
optimizador Adam
es un método de descenso de gradiente estocástico que se basa en la estimación 
adaptativa de momentos de primer y segundo orden.
'''


def preparar_modelo():
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=['accuracy'])
    tensorboard = TensorBoard(log_dir='logs')
    checkpoint = ModelCheckpoint(
        "effnet.h5", monitor="val_accuracy", save_best_only=True, mode="auto", verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3, patience=2, min_delta=0.001,
                                  mode='auto', verbose=1)
    return (tensorboard, checkpoint, reduce_lr)


'''
funcion entrenar luego de haber configurado las diferentes capas nuestra imagen inicia a un proceso
de entrenamiento , nos ofrece un archivo .h5 que es un archivo de metadatos el cual
ya es un modelo entrenado para realizar pruebas 
'''


def entrenar():
    (tensorboard, checkpoint, reduce_lr) = preparar_modelo()
    history = model.fit(X_train, y_train, validation_split=0.1, epochs=12, verbose=1, batch_size=32,
                        callbacks=[tensorboard, checkpoint, reduce_lr])
    return history


effnet = descarga_modelo()
model = effnet.output

'''
Predicción
He usado la función argmax ya que cada fila de la matriz 
de predicción contiene cuatro valores para las etiquetas 
respectivas. El valor máximo que está en cada fila representa
 la salida prevista de los 4 resultados posibles.
Entonces, con argmax, puedo averiguar el índice asociado
 con el resultado predicho.
'''


def principal_entrenamiento(request):
    carga_de_imagenes()
    convertir_imagenes()
    muestreo_imagenes()
    array_imagen()
    entrenamiento_test()
    entrenando_labels()
    entrenando_labels_prueba()
    descarga_modelo()
    informacion = instructor_modelo()

    return JsonResponse({
        'files': log_carga_imagenes,
        'informacion': informacion,
        "confu": "http://127.0.0.1:8000/static/assets/img/confu.png",
        "matrix": "http://127.0.0.1:8000/static/assets/img/matrices.png"
    })
# =================Segmentacion===================


def preprocess(img):
    # use the pre processing function of ResNet50
    img = preprocess_input(img)

    # expand the dimension
    return np.expand_dims(img, 0)


def get_activations_at(input_image, i, resnet_50):
    # index the layer
    out_layer = resnet_50.layers[i]

    # change the output of the model
    model = tf.keras.models.Model(
        inputs=resnet_50.inputs, outputs=out_layer.output)

    # return the activations
    return model.predict(input_image)


def postprocess_activations(activations):

    # using the approach in https://arxiv.org/abs/1612.03928
    output = np.abs(activations)
    output = np.sum(output, axis=-1).squeeze()

    # resize and convert to image
    output = cv2.resize(output, (224, 224))
    output /= output.max()
    output *= 255
    return 255 - output.astype('uint8')


def apply_heatmap(weights, img):
    # generate heat maps
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, 0.7, img, 0.3, 0)
    return heatmap


def plot_heatmaps(rng, input_image, img, resnet_50):
    level_maps = None

    # given a range of indices generate the heat maps
    for i in rng:
        activations = get_activations_at(input_image, i, resnet_50)
        weights = postprocess_activations(activations)
        heatmap = apply_heatmap(weights, img)
        if level_maps is None:
            level_maps = heatmap
        else:
            level_maps = np.concatenate([level_maps, heatmap], axis=1)
    plt.figure(figsize=(15, 15))
    plt.axis('off')

    plt.imshow(level_maps)
    return plt


def inicio(ruta):
    resnet_50 = tf.keras.applications.ResNet50(input_shape=(224, 224, 3),
                                               include_top=False,
                                               weights='imagenet')
    for i, addres in enumerate(glob.glob(ruta)):
        img = cv2.imread(addres)
        img = cv2.resize(img, (224, 224))
        #ax = plt.imshow(img)
    input_image = preprocess(img)
    return (img, input_image, resnet_50)


def mostrar_imagen(request, foto):

    (img, input_image, resnet_50) = inicio(
        "./media/resonancias/"+foto)
    px = plot_heatmaps(range(164, 169), input_image, img, resnet_50)

    fig = px
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    string = base64.b64encode(buf.read())
    uri = urllib.parse.quote(string)

    return JsonResponse({'data': uri})
