from pathlib import Path
from typing import List, Union

def remove_folder(path: Union[Path,List[Path]],recursive: bool = False) -> bool:
    if isinstance(path,list):
        for p in path:
            remove_folder(p)
    elif path.exists():
        for child in path.iterdir():
            if child.is_dir():
                try:
                    child.rmdir()
                except:
                    remove_folder(child)
            else:
                child.unlink(child)
        path.rmdir()
        if recursive:
            return True
    print(f'[FOLDERS] Removidos: {path}')
    return True

def create_folders(
    name: Path,
    childrens: Union[List[str],str, None] = None
) -> bool:

    name.mkdir(exist_ok=True)

    if childrens is not None:
        if isinstance(childrens, list):
            for child in childrens:
                create_folders(name / Path(child))
        else:
            create_folders(Path(childrens))

    print(f'[FOLDERS] Criadas {str(name)}/{str(childrens)}')

    return True
