"""Run an experiment."""

import logging
import sys
import os
import yaml
import importlib
import pprint

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)

class Config():
    '''Configuration Class'''

    def __init__(self, config_filepath=None):
        if config_filepath is None:
            self.config_filepath = "./config/config.yaml"
        else:
            self.config_filepath = config_filepath


    def main(self, verbose=False):
        """Load the configuration and print it."""
        cfg = self.load_cfg()

        # Add modules to system path:
        modules = cfg['modules']            
        for k, v in modules.items():
            sys.path.append(os.path.dirname(v))

        # Print the configuration - just to make sure that you loaded what you
        # wanted to load
        if verbose:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(cfg)           
        

    def load_cfg(self):
        """
        Load a YAML configuration file.

        Parameters
        ----------
        config_filepath : str

        Returns
        -------
        cfg : dict
        """
        # Read YAML experiment definition file
        with open(self.config_filepath, 'r') as stream:
            cfg = yaml.load(stream)
        cfg = self.make_paths_absolute(os.path.dirname(self.config_filepath), cfg)
        return cfg


    def make_paths_absolute(self, dir_, cfg):
        """
        Make all values for keys ending with `_module` absolute to dir_.

        Parameters
        ----------
        dir_ : str
        cfg : dict

        Returns
        -------
        cfg : dict
        """
        for key in cfg.keys():
            if key.endswith("_module"):
                cfg[key] = os.path.join(dir_, cfg[key])
                cfg[key] = os.path.abspath(cfg[key])
                if not os.path.isfile(cfg[key]):
                    logging.error("%s does not exist.", cfg[key])
            if type(cfg[key]) is dict:
                cfg[key] = self.make_paths_absolute(dir_, cfg[key])
        return cfg


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    config = Config(args.filename)
    config.main()


