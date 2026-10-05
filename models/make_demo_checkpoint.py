"""Build a small, explicitly MOCK checkpoint; never reads or writes survey rows."""
import argparse
from models.train_cvae import main as train


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', required=True)
    a = p.parse_args(argv)
    return train(['--mock', '--demo-profiles', '--epochs', '2', '--max-rows', '4000', '--cpu', '--out-dir', a.out_dir])


if __name__ == '__main__':
    main()
