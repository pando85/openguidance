

VERSION = "0.0.1"
__version__ = VERSION

import logging

from guidancelib import configuration as cfg

config = None

from guidancelib import gps
from guidancelib import guidance
from guidancelib import pathprovider

import logging
logging.root.setLevel(logging.INFO)

def version():
    return """OpenGuidance {cm_version}
open source agricultural GPS guidance
Copyright (c) 2015 Alexander Gil Casas""".format(cm_version=VERSION)

def info():
    return "Info" 

def setup_config(override_dict=None):
    """ Updates the internal configuration using the following hierarchy:
        override_dict > file_config > default_config

        Notifies the user if there are new or deprecated configuration keys.

        See :mod:`~guidancelib.configuration`.
    """
    defaults = cfg.from_defaults()
    filecfg = cfg.from_configparser(pathprovider.configuration_file())
    custom = defaults.replace(filecfg, on_error=logging.error)
    if override_dict:
        custom = custom.replace(override_dict, on_error=logging.error)
    global config
    config = custom
    _notify_about_config_updates(defaults, filecfg)

def _notify_about_config_updates(default, known_config):
    """check if there are new or deprecated configuration keys in
    the config file
    """
    new = []
    deprecated = []
    transform = lambda s: '[{0}]: {2}'.format(*(s.partition('.')))

    for property in cfg.to_list(default):
        if property.key not in known_config and not property.hidden:
            new.append(transform(property.key))
    for property in cfg.to_list(known_config):
        if property.key not in default:
            deprecated.append(transform(property.key))

    if new:
        logging.info(('''New configuration options available:
                    %s
                Using default values for now.'''),
              '\n\t\t\t'.join(new))
    if deprecated:
        logging.info(('''The following configuration options are not used anymore:
                    %s'''),
              '\n\t\t\t'.join(deprecated))
    if new or deprecated:
        logging.info(('Start with --newconfig to generate a new default config'
                ' file next to your current one.'))

def create_default_config_file(path):
    """ Creates or overwrites a default configuration file at `path` """
    cfg.write_to_file(cfg.from_defaults(), path)
    logging.info(('Default configuration file written to %(path)r'), {'path': path})
