import argparse
from pathlib import Path
import shutil
from src.train import Train
from config import Configuration as config
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'project_name',
        help="name to be shown in tensorboard/under which model will be saved",
    )
    parser.add_argument(
        '--test_session',
        help='test session will load less images',
        action='store_true',
    )
    args = parser.parse_args()
    config.just_testing = args.test_session
    if os.path.exists(os.path.join(config.work_dir, args.project_name)):
        shutil.rmtree(os.path.join(config.work_dir, args.project_name))

    trainer = Train(config=config, name=args.project_name)
    trainer()


if __name__ == '__main__':
    main()
