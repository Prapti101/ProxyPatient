"""Private handover assembly. Test metadata comes only from an existing manifest."""
import argparse
import json
import shutil
from pathlib import Path
from models.common import read_parquet, sha256_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args(argv)
    directory, output = Path(args.data_dir), Path(args.out_dir)
    if directory.resolve() == output.resolve():
        raise ValueError('Handover output must differ from source')
    with (directory/'MANIFEST.json').open() as f:
        entries = json.load(f)['manifest']
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for original in entries:
        name = original['file']
        if Path(name).name != name:
            raise ValueError('Manifest filenames must be basenames')
        source = directory/name
        entry = dict(original)
        if 'test' not in name.lower():
            entry['sha256'] = sha256_file(str(source))
            if name.endswith('.parquet'):
                frame = read_parquet(str(source))
                entry.update(shape=list(frame.shape), columns=list(frame.columns))
        elif not all(k in entry for k in ('sha256', 'shape', 'columns')):
            raise ValueError('Existing test manifest metadata required; test contents are never decoded')
        shutil.copy2(source, output/name)
        manifest.append(entry)
    write_json({'manifest': manifest}, str(output/'MANIFEST.json'))
    (output/'README.md').write_text('PRIVATE licensed handover. Never commit data or weights. Test metadata copied from existing manifest.\n')
    return manifest


if __name__ == '__main__':
    main()
