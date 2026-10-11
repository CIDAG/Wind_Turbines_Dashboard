"""Extract Penmanshiel CSVs from the Zenodo ZIP directly into data/."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


PENMANSHIEL_CSV = re.compile(r"(?:Turbine_Data|Status)_Penmanshiel_.*\.csv", re.IGNORECASE)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def extract_penmanshiel(zip_path: Path, data_dir: Path) -> int:
    """Flatten the ZIP's Penmanshiel CSVs into data_dir without overwriting files."""
    zip_path = zip_path.resolve()
    data_dir = data_dir.resolve()

    with ZipFile(zip_path) as archive:
        members = [
            member
            for member in archive.infolist()
            if not member.is_dir()
            and PENMANSHIEL_CSV.fullmatch(PurePosixPath(member.filename).name)
        ]
        if not members:
            raise ValueError("O ZIP nao contem CSVs Penmanshiel esperados.")

        names = [PurePosixPath(member.filename).name for member in members]
        if len(names) != len({name.casefold() for name in names}):
            raise ValueError("O ZIP contem nomes de CSV duplicados em pastas diferentes.")

        existing = [name for name in names if (data_dir / name).exists()]
        if existing:
            raise FileExistsError(f"Arquivos ja existem em data/: {', '.join(existing[:3])}")

        data_dir.mkdir(parents=True, exist_ok=True)
        for member, name in zip(members, names):
            target = data_dir / name
            try:
                with archive.open(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            except Exception:
                target.unlink(missing_ok=True)
                raise

    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_path", type=Path, help="Caminho para datasets2020-penmanshiel.zip")
    arguments = parser.parse_args()
    count = extract_penmanshiel(arguments.zip_path, PROJECT_ROOT / "data")
    print(f"{count} arquivos Penmanshiel extraidos diretamente em {PROJECT_ROOT / 'data'}")


if __name__ == "__main__":
    main()
