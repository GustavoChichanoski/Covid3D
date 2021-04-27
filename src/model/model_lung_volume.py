from typing import Tuple
from typing_extensions import Concatenate
from pathlib import Path
from tensorflow.python.keras import Model
from tensorflow.python.keras.engine.input_layer import Input
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.optimizers import Adamax
from src.output.save_csv import save_csv
from src.model.unet_function import get_callbacks, metrics

class LungVolume:

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (256, 256, 1),
        activation: str = 'linear',
        filter_root: int = 32
    ) -> None:
        super(LungVolume, self).__init__()
        self.input_size = input_size
        self.activation = activation
        self.filter_root = filter_root
        self._lazy_model = None
        self._lazy_callbacks = None
        self._lazy_metrics = None

    @property
    def callbacks(self):
        if self._lazy_callbacks is None:
            self._lazy_callbacks = get_callbacks()
        return self._lazy_callbacks
    
    @property
    def metrics(self):
        if self._lazy_metrics is None:
            self._lazy_metrics = metrics()
        return self._lazy_metrics


    @property
    def model(self) -> Model:
        if self._lazy_model is None:
            input = Input(shape=self.input_size)
            layer = input
            kernel_size = (3, 3)
            depth = int(self.input_size[1] / 8)
            for i in range(depth):
                filters = (2**i) * self.filter_root
                params = {'kernel_size': kernel_size, 'padding': 'same'}
                layer = Conv2D(
                    filters=filters,
                    name=f'CV_{i}',
                    **params
                )(layer)
                layer = Activation(
                    activation=self.activation,
                    name=f'Act_{i}'
                )(layer)
            layer = Flatten()(layer)
            lung_pixel = Input(shape=(1,))
            layer = Concatenate(axis=1)([layer, lung_pixel])
            layer = Dense(filters=self.filter_root,activation='relu')(layer)
            self._lazy_model = Model(input, layer, name="Volume")
        return self._lazy_model

    def compile(
        self,
        loss: str = 'mean_squared_error',
        lr: float = 1e-2,
        **params
    ) -> None:
        opt = Adamax(learning_rate=lr)
        compile_params = {'optimizer':opt,'metrics':self.metrics, **params}
        self.model.compile(loss=loss, **compile_params)
        return None

    def save(
        self,
        history=None,
        model: str = 'Volume',
        name: str = None,
        metric: str = 'val_loss',
        parent: Path = Path('./'),
        verbose: bool = False
    ) -> str :
        if name is not None:
            file = parent / name
            self.model.save(f'{file}.hdf5',overwrite=True)
            self.model.save_weights(f'{file}_weights.hdf5', overwrite=True)
        else:
            value = 100.00
            file = 'model.hdf5'
            if history is not None:
                value = history[metric][-1] * 100
                history_path = parent / 'history' / f'history_{model}_{value}.hdf5'
                save_csv(value=history, name=history_path, verbose=verbose)
            file = parent / 'weights' / f'{model}_{metric}_{value:.02f}.hdf5'
            self.model.save_weights(file,overwrite=True)
            if verbose:
                print(f'[MODEL] Pesos salvos em: {file}')
            return file
        return None