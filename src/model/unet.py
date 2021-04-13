from src.data.generator import Generator
from keras.metrics import Metric
from src.model.unet_function import dice_coef_loss, get_callbacks, metrics, unet_conv, up_conct
from src.output.save_csv import save_csv
from typing import List, Tuple
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.optimizers import Adamax
from pathlib import Path


class UNet:

    def __init__(
        self,
        input_size: Tuple[int, int, int] = (256, 256, 1),
        filter_root: int = 32,
        activation: str = 'relu',
        final_activation: str = 'sigmoid',
        depth: int = 5,
        n_class: int = 1
    ) -> None:
        super(UNet, self).__init__()
        self.input_size = input_size
        self.filter_root = filter_root
        self.activation = activation
        self.final_activation = final_activation
        self.depth = depth
        self.n_class = n_class

        self._lazy_callbacks = None
        self._lazy_metrics = None
        self._lazy_model = None

    @property
    def callbacks(self):
        if self._lazy_callbacks is None:
            self._lazy_callbacks = get_callbacks()
        return self._lazy_callbacks

    @property
    def metrics(self) -> List[Metric]:
        if self._lazy_metrics is None:
            self._lazy_metrics = metrics()
        return self._lazy_metrics

    def save(
        self,
        history=None,
        model: str = 'UNet',
        name: str = None,
        metric: str = 'val_f1',
        parent: Path = Path('./'),
        verbose: bool = False
    ) -> str:
        if name is not None:
            file = parent / name
            self.model.save(f'{file}.hdf5', overwrite=True)
            self.model.save_weights(f'{file}_weights.hdf5', overwrite=True)
        else:
            value = 100.00
            file = 'model.hdf5'
            if history is not None:
                value = history[metric][-1] * 100
                history_path = parent / 'historys' / f'history_{model}_{value}'
                save_csv(value=history, name=history_path, verbose=verbose)
            file = parent / 'weights' / f'{model}_{metric}_{value:.02f}.hdf5'
            self.model.save_weights(file, overwrite=True)
            if verbose:
                print(f'[MODEL] Pesos salvos em: {file}')
            return file
        return None

    def fit(
        self,
        train_generator: Generator,
        val_generator: Generator,
        **params
    ):
        history = self.model.fit(
            x=train_generator,
            validation_data=val_generator,
            callbacks=self.callbacks,
            **params
        )
        return history

    def compile(
        self,
        loss: str = 'dice_coef',
        lr: float = 1e-3,
        **params
    ) -> None:
        opt = Adamax(learning_rate=lr)
        compile_params = {'optimizer': opt, 'metrics': self.metrics, **params}
        if loss == 'dice_coef':
            self.model.compile(loss=dice_coef_loss, **compile_params)
        else:
            self.model.compile(loss=loss, **compile_params)
        return None

    @property
    def model(self) -> Model:
        if self._lazy_model is None:
            kernel_size = (3, 3)
            store_layers = {}
            inputs = Input(self.input_size)
            first_layers = inputs
            params = {'kernel_size': kernel_size,
                      'activation': self.activation}
            for i in range(self.depth):
                filters = (2**i) * self.filter_root
                layer = unet_conv(
                    layer=first_layers,
                    filters=filters,
                    depth=i,
                    name="Down",
                    **params
                )
                if i < self.depth - 1:
                    store_layers[str(i)] = layer
                    first_layers = MaxPooling2D(
                        pool_size=(2, 2),
                        padding='same',
                        name=f'MaxPooling{i}_0'
                    )(layer)
                else:
                    first_layers = layer
            for i in range(self.depth-2, -1, -1):
                filters = (2**i) * self.filter_root
                connection = store_layers[str(i)]
                layer = up_conct(
                    layer=first_layers,
                    connection=connection,
                    depth=self.depth - i
                )
                layer = unet_conv(
                    layer=layer,
                    filters=filters,
                    depth=self.depth - i,
                    name="Up",
                    **params
                )
                first_layers = layer
            layer = Dropout(0.33, name='Drop_1')(layer)
            outputs = Conv2D(
                filters=self.n_class,
                kernel_size=(1, 1),
                padding='same',
                activation=self.final_activation,
                name='output'
            )(layer)
            self._lazy_model = Model(inputs, outputs, name="UNet")
        return self._lazy_model
