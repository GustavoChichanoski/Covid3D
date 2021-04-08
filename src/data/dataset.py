from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class Dataset:
    """Gera o dataset pronto para ser lido pelo keras.
        Args:
        -----
            path_data (Path): Caminho para as imagens de CTs
            dimension_original (int): dimensão da imagem original
            dimension_cut (int): dimensão do corte da imagem
    """

    path_data: Path

    dimension_original: int = 1024
    dimension_cut: int = 224
    channels: int = 3
    
    csv_column: str = 'img_id'
    ct: Path = Path('CTs')

    _lazy_label_names: Optional[List[Path]] = None
    _lazy_files_in_folder: Optional[List[Path]] = None
    _lazy_x: Optional[List[Path]] = None
    _lazy_y: Optional[Any] = None
    _lazy_csv: Optional[Any] = None
    _lazy_ct_2d: Optional[Any] = None
    _lazy_ct_2d_mask: Optional[Any] = None

    @property
    def ct_2d(self) -> Path:
        if self._lazy_ct_2d is None:
            self._lazy_ct_2d = self.ct / '2d_images'
        return self._lazy_ct_2d
    
    @property
    def ct_2d_mask(self) -> Path:
        if self._lazy_ct_2d_mask is None:
            self._lazy_ct_2d_mask = self.ct / '2d_masks'
        return self._lazy_ct_2d_mask

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
            data = pd.read_csv(self.path_data)[self.csv_column]
            if self.csv_column == 'img_id':
                self._lazy_y = [self.ct_2d_mask / filename for filename in data]
            elif self.csv_column == 'lung_area_px':
                self._lazy_y = [int(i) for i in data]
            else:
                self._lazy_y = [float(i) for i in data]
        return self._lazy_y

    @property
    def x(self) -> List[str]:
        if self._lazy_x is None:
            data = pd.read_csv(self.path_data)['img_id']
            self._lazy_x = [self.ct_2d / filename for filename in data]
        return self._lazy_x

    def partition(self,
                  val_size: float = 0.2) -> Tuple[Tuple[Any, Any], Tuple[Any, Any]]:
        """ Retorna a entrada e saidas dos keras.

            Args:
            -----
                val_size (float, optional): Define o tamanho da validacao.
                                            Defaults to 0.2.
            Returns:
            --------
                (test), (val): Saida para o keras.
        """
        # t : train - v : validation
        train_in, val_in, train_out, val_out = train_test_split(
            self.x,
            self.y,
            test_size=val_size,
            shuffle=True
        )
        train, val = (train_in, train_out), (val_in, val_out)
        return train, val