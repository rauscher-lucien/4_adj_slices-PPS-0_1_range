import os
import sys
import argparse
sys.path.append(os.path.join(".."))

import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp formatlog_file = open('logfile.log', 'w', buffering=1)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)





from utils import *
from train import *


def main():

    ## parser

    # Check if the script is running on the server by looking for the environment variable
    if os.getenv('RUNNING_ON_SERVER') == 'true':
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Process data directory.')
        parser.add_argument('--data_dir', type=str, help='Path to the data directory')
        parser.add_argument('--project_name', type=str, help='Name of the project')
        parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                            help='Flag to continue training: "on" or "off" (default: "off")')
        parser.add_argument('--load_epoch', type=int, default=1, 
                            help='Epoch number from which to continue training (default: 1)')
    

        # Parse arguments
        args = parser.parse_args()

        # Now you can use args.data_dir as the path to your data
        data_dir = args.data_dir
        project_name = args.project_name 
        train_continue = args.train_continue
        load_epoch = args.load_epoch
        project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', '4_adj_slices-PPS-0_1_range')

        print(f"Using data directory: {data_dir}")
        print(f"Project name: {project_name}")
        print(f"Train continue: {train_continue}")
        print(f"Load epoch: {load_epoch}")
    else:

        data_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'data', 'big_data_small')
        project_name = 'big_data_small-test-1'
        train_continue = 'off'
        load_epoch = 1
        project_dir = os.path.join('C:\\', 'Users', 'rausc', 'Documents', 'EMBL', 'projects', '4_adj_slices-PPS-0_1_range')

        print(f"Not running on server, using default data directory: {data_dir}")
        print(f"Default project name: {project_name}")


    results_dir, checkpoints_dir = create_result_dir(project_dir, project_name)

    data_dict = {}

    data_dict['results_dir'] = results_dir
    data_dict['checkpoints_dir'] = checkpoints_dir
    data_dict['data_dir'] = data_dir

    data_dict['num_epoch'] = 300
    data_dict['batch_size'] = 8
    data_dict['lr'] = 10e-5

    data_dict['num_freq_disp'] = 50
    data_dict['num_freq_save'] = 1

    data_dict['train_continue'] = train_continue
    data_dict['load_epoch'] = load_epoch


    TRAINER = Trainer(data_dict)
    TRAINER.train()


if __name__ == '__main__':
    main()


