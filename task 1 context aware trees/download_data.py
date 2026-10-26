"""Download datasets 0, 1 and 2 with their matching labels."""
import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE_URL = ('https://github.com/dsa4262-yellow-2026/dsa4262-m6a-context-aware-trees/'
               'releases/download/training-data-v1/training-data-v1.zip')
INPUT_NAMES = (
    'data0/dataset0.json.gz', 'data0/data.info.labelled',
    'data1/dataset1.json.gz', 'data1/data.info',
    'data2/dataset2.json.gz', 'data2/data.info',
)


def prepare(output, archive=None):
    output = Path(output).resolve()
    missing = [name for name in INPUT_NAMES if not (output / name).is_file()]
    if not missing:
        print(f'Training inputs are already available: {output}', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='training-download-', dir=output) as directory:
        work = Path(directory)
        if archive is None:
            archive = work / 'training-data.zip'
            print('Downloading the training inputs (328 MB).', flush=True)
            with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as source, archive.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024**2)
        with zipfile.ZipFile(archive) as bundle:
            if set(bundle.namelist()) != set(INPUT_NAMES) or len(bundle.namelist()) != len(INPUT_NAMES):
                raise ValueError('The archive must contain the six documented training files.')
            for name in missing:
                staged = work / name
                staged.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(name) as source, staged.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1024**2)
            for name in missing:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                (work / name).replace(target)
    print(f'Training inputs saved in: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE / 'inputs/training',
                        help='Destination containing data0/, data1/ and data2/.')
    parser.add_argument('--archive', type=Path,
                        help='Use a downloaded training-data-v1.zip instead of the network.')
    args = parser.parse_args()
    try:
        prepare(args.output, args.archive)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, f'Training-data download failed: {error}\n')


if __name__ == '__main__':
    main()
