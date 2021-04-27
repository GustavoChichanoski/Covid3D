from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class Dataset:
    """Gera o dataset pronto para ser lido pelo keras.

        >>> dataset = Dataset('imagens.csv', 'img_id')

        Args:
        -----
            csv (Path): Caminho para o csv dos caminhos das imagens
            csv_column (str): Nome da coluna contendo os nomes das imagens
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão do corte da imagem
    """
    csv: Path
    csv_column: str = 'img_id'

    dimension_original: int = 1024
    dimension_cut: int = 224
    channels: int = 3
    
    path_x: Path = Path('CTs')
    path_y: Optional[Path]= None

    _lazy_label_names: Optional[List[Path]] = None
    _lazy_files_in_folder: Optional[List[Path]] = None
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None

    @property
    def files_in_folders(self) -> List[List[Path]]:
        """Retorna os arquivos contidos nas pastas

            Returns:
                List[List[Path]]: Lista contendo os arquivos da
                pasta Original
        """
        if self._lazy_files_in_folder is None:
            self._lazy_files_in_folder = [list(folder.iterdir()) for folder in self.path_data]
        return self._lazy_files_in_folder
    
    @property
    def y(self) -> List[str]:
        if self._lazy_y is None:
            if self.path_y is not None:
                data = pd.read_csv(self.csv)['img_id']
                self._lazy_y = [self.path_y / filename for filename in data]
            else:
                data = pd.read_csv(self.csv)[self.csv_column]
                self._lazy_y = [float(value) for value in data]
        return self._lazy_y

    @property
    def x(self) -> List[str]:
        """Retorna os caminhos das imagens a serem lidos pelo modelo

        >>> dataset = Dataset(path)
        >>> dataset.x

        Args:
        -----
            None

        Returns:
        --------
            List[str]: Caminhos das imagens
        """        
        if self._lazy_x is None:
            data = pd.read_csv(self.csv)['img_id']
            self._lazy_x = [self.path_x / filename for filename in data]
        return self._lazy_x

    def partition(
        self,
        val_size: float = 0.2,
        test: bool = False
    ) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """ Retorna a entrada e saidas dos keras.

            >>> dataset = Dataset(path)
            >>> test_x, test_y = dataset.partition(0.2, True)

            Args:
            -----
                val_size (float, optional): Define o tamanho da validacao.
                                            Defaults to 0.2.
                test (bool, optional): Partição a ser criada é para teste ?.
                                       Defaults to False
            Returns:
            --------
                (test), (val): Saida para o keras.
        """
        # t : train - v : validation
        if test:
            x, y = self.x[0:4], self.y[0:4]
        else:
            x, y = self.x, self.y

        train_in, val_in, train_out, val_out = train_test_split(
            x,
            y,
            test_size=val_size,
            shuffle=True
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val