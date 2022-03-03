from typing import Tuple, Optional, Union, Dict
from pickle import load as pickle_load
from pathlib import Path

from torch.utils.data import Dataset
import numpy as np
from utils import get_files_from_dir_with_pathlib

__docformat__ = 'reStructuredText'
__all__ = ['IrmasDataset']


class IrmasDataset(Dataset):

        def __init__(self,
                     data_dir: Union[str, Path],
                     key_features: Optional[str] = 'features',
                     key_class: Optional[str] = 'class') \
                -> None:

            super().__init__()
            self.key_features = key_features
            self.key_class = key_class

            data_path = Path(data_dir)
            self.files = get_files_from_dir_with_pathlib(data_path)

            for i, a_file in enumerate(self.files):
                self.files[i] = self._load_file(a_file)

        @staticmethod
        def _load_file(file_path: Path)\
                -> Dict[np.ndarray, np.ndarray]:
            """Loads a file using pathlib.Path

            :param file_path: File path.
            :type file_path: pathlib.Path
            :return: The file.
            :rtype: dict[str, int|numpy.ndarray]
            """
            with file_path.open('rb') as f:
                return pickle_load(f)

        def __len__(self) \
                -> int:
            """Returns the lenght of the dataset."""

            return len(self.files)

        def __getitem__(self,
                        item: int) \
                -> Tuple[np.ndarray, np.ndarray]:
            the_item: Dict[str, Union[int, np.ndarray]] = self.files[item]

            return the_item[0], the_item[1]

# EOF
