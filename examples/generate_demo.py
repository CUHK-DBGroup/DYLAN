#!/usr/bin/env python3
"""Generate small inputs for the README commands; no external datasets are needed."""
import argparse
from pathlib import Path
import struct


def write_vectors(path, rows):
    with path.open('wb') as out:
        for row in rows:
            out.write(struct.pack('<i' + 'f' * len(row), len(row), *row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/demo'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    write_vectors(args.output / 'base.fvecs', [[float(i), float(i % 3)] for i in range(12)])
    write_vectors(args.output / 'query.fvecs', [[0., 0.], [2., 2.]])
    write_vectors(args.output / 'new.fvecs', [[0.1, 0.], [0.2, 0.]])
    files = {
        'base.csv': '1\n2\n1,2\n\n' * 3,
        'query.csv': '1\n2\n',
        'new.csv': '1,10001\n10001\n',
        'new_ids.txt': '100\n101\n',
        'new_query.csv': '10001\n10001\n',
        'delete_ids.txt': '100\n',
    }
    for name, content in files.items():
        (args.output / name).write_text(content, encoding='utf-8')
    print(f'Demo inputs written to {args.output}')


if __name__ == '__main__':
    main()
