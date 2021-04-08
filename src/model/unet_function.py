import importlib

from src.model.metrics.f1_score import F1score
from typing import List, Tuple
from keras.layers import Layer
from keras.layers import Concatenate
from keras.layers import UpSampling2D
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.metrics import Metric
from keras.metrics import TruePositives
from keras.metrics import TrueNegatives
from keras.metrics import FalseNegatives
from keras.metrics import FalsePositives
from keras.metrics import BinaryAccuracy
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN
from keras import backend as K

def get_callbacks() -> List[Callback]:
    """
        Retorna a lista callbacks do modelo
        Args:
        -----
            weight_path: Caminho para salvar os checkpoints
        Returns:
        --------
            (list of keras.callbacks): lista dos callbacks
    """
    # Salva os pesos dos modelo para serem carregados
    # caso o monitor não diminua
    check_params = {
        'monitor': 'val_loss', 'verbose': 1, 'mode': 'min',
        'save_best_only': True, 'save_weights_only': True
    }
    checkpoint = ModelCheckpoint('./checkpoints/', **check_params)

    # Reduz o valor de LR caso o monitor nao diminuia
    reduce_params = {
        'factor': 0.5, 'patience': 3, 'verbose': 1,
        'mode': 'min', 'min_delta': 1e-3,
        'cooldown': 2, 'min_lr': 1e-8
    }
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', **reduce_params)

    # Parada do treino caso o monitor nao diminua
    stop_params = {'mode':'min', 'restore_best_weights':True, 'patience':40}
    early_stop = EarlyStopping(monitor='val_f1', **stop_params)

    # Termina se um peso for NaN (not a number)
    terminate = TerminateOnNaN()

    # Habilita a visualizacao no TersorBoard
    # tensorboard = TensorBoard(log_dir="./logs")

    # Armazena os dados gerados no treinamento em um CSV
    # csv_logger = CSVLogger('./logs/trainig.log', append=True)

    # Vetor a ser passado na função fit
    # callbacks = [checkpoint, early_stop, reduce_lr, terminate, tensorboard, csv_logger]
    callbacks = [checkpoint, early_stop, reduce_lr, terminate]
    return callbacks

def unet_conv(
    layer: Layer,
    filters: int = 32,
    kernel_size: Tuple[int, int] = (3, 3),
    activation: str = 'relu',
    depth: int = 0
) -> Layer:
    for i in range(2):
        params = {'filters': filters, 'kernel_size': kernel_size, 'padding':'same'}
        layer = Conv2D(name=f"Conv{depth}_{i}",**params)(layer)
        layer = Activation(activation=activation, name=f"Act{depth}_{i}")(layer)
    return layer

def up_conct(layer: Layer, connection: Layer, depth:int = 0) -> Layer:
    layer = UpSampling2D(name=f'UpSampling{depth}_1')(layer)
    layer = Concatenate(axis=-1, name=f'UpConcatenate{depth}_1')([layer, connection])
    return layer

def metrics() -> List[Metric]:
    tp = TruePositives(name='tp')
    tn = TrueNegatives(name='tn')
    fp = FalseNegatives(name='fp')
    fn = FalsePositives(name='fn')
    ba = BinaryAccuracy(name='ba')
    f1 = F1score(name='f1')
    return [tp,tn,fp,fn,ba,f1]

def dice_coef(y_true, y_pred):
    ''' Dice Coefficient
    Project: BraTs   Author: cv-lee   File: unet.py    License: MIT License
    Args:
        y_true (np.array): Ground Truth Heatmap (Label)
        y_pred (np.array): Prediction Heatmap
    Returns:
        (np.array): Calcula a porcentagem de acerto da rede neural
    '''

    class_num = 1

    for class_now in range(class_num):

        # Converte y_pred e y_true em vetores
        y_true_f = K.flatten(y_true[:, :, :, class_now])
        y_pred_f = K.flatten(y_pred[:, :, :, class_now])

        # Calcula o numero de vezes que
        # y_true(positve) é igual y_pred(positive) (tp)
        intersection = K.sum(y_true_f * y_pred_f)
        # Soma o número de vezes que ambos foram positivos
        union = K.sum(y_true_f) + K.sum(y_pred_f)
        # Smooth - Evita que o denominador fique muito pequeno
        smooth = K.constant(1e-6)
        # Calculo o erro entre eles
        num = (K.constant(2)*intersection + 1)
        den = (union + smooth)
        loss = num / den

        if class_now == 0:
            total_loss = loss
        else:
            total_loss = total_loss + loss

    total_loss = total_loss / class_num

    return total_loss

def dice_coef_loss(y_true, y_pred):
    accuracy = dice_coef(y_true, y_pred)
    return accuracy

# def callbacks() -> List[Callback]: