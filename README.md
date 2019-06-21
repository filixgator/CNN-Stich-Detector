# Medtronic Aortic Heart Valve Stitch Detector
## Objetivo
El presente proyecto consiste en un sistema capaz de identificar las puntadas que forman las costuras de una válvula cardiovascular, como base para posibilitar la detección de errores en la fabricación de la misma, implementando herramientas de visión artificial.
## Tecnología
Se hace uso de una implementación open source de una Mask R-CNN, la cual puede consultarse en https://github.com/matterport/Mask_RCNN
## En este Repositorio
* Mask_R_CNN_Colab_Notebook.ipynb
  * Google Colab Notebook con el entrenamiento de la red, y los resultados iniciales
* Executable_GUI.py
  * Código en Python de la interfaz de usuario, requiere descargar el modelo con los valores de la red: https://drive.google.com/drive/folders/1lJJsvQHN2uDUdhHP8g0uBbIso_a9dmh8?usp=sharing
## Resultados
* Imágenes muestra
   * ![img1](/assets/2.png)
* Preprocesado
   * ![img2](/assets/2sharp.png)
* Segmentación
   * ![img3](/assets/result2.png)
