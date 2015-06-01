import os
import sys

config_folder_name = 'openguidance'
config_file_name = 'openguidance.conf'

def get_config_path():
    #TODO: change config path with arg
    configpath = ''
    if sys.platform.startswith('linux'):  # linux
        if 'XDG_CONFIG_HOME' in os.environ:
            configpath = os.path.join(os.environ['XDG_CONFIG_HOME'], config_folder_name)
        else:
            configpath = os.path.join(os.path.expanduser('~'), '.config', config_folder_name)
    elif sys.platform.startswith('win'): #windows
        configpath = os.path.join(os.environ['APPDATA'],config_folder_name)
    elif sys.platform.startswith('darwin'): #osx
        configpath = os.path.join(os.path.expanduser('~'), 'Library', 'Application Support', config_folder_name)

    if not configpath:
        configpath = fallback_path()
    assure_folder_exists(configpath)
    return configpath

def fallback_path():
    return os.path.join(os.path.expanduser('~'), '.openguidance')

def assure_folder_exists(folder, subfolders=['']):
    for subfolder in subfolders:
        dirpath = os.path.join(folder, subfolder)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

def configuration_file():
    return os.path.join(get_config_path(), config_file_name)

def configuration_file_exists():
    return os.path.exists(configuration_file())