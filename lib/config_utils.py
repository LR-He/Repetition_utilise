import logging
import os
import sys
from typing import Optional

import prodict
import yaml
from omegaconf import DictConfig, OmegaConf
from prodict import Prodict


class _PrettySafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))


_PrettySafeLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    _PrettySafeLoader.construct_python_tuple)


def resolve_tuple(*args):
    """Resolves a value in a config file with a structure like ${tuple:1,2} to a tuple (1, 2)."""
    return tuple(args)


OmegaConf.register_new_resolver('tuple', resolve_tuple)


def print_config(config: DictConfig | prodict.Prodict | str, logger: Optional[logging.Logger] = None) -> None:
    """
    Prints a yaml configuration file to the console.

    Args:
        config:      string or dict, path of the yaml file or previously imported configuration file.
        logger:      logger instance.
    """

    if isinstance(config, Prodict):
        config = config.to_dict(is_recursive=True)
    elif isinstance(config, str):
        config = read_config(config)
    elif isinstance(config, DictConfig):
        config = OmegaConf.to_container(config)

    if logger:
        logger.info(yaml.dump(config, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True))
    else:
        yaml.dump(config, sys.stdout, indent=4, default_flow_style=False, sort_keys=False, allow_unicode=True)


"""
这段代码定义了一个函数 read_config，它的作用是从一个 YAML 配置文件中读取配置信息，并返回一个包含配置的 Python 字典，
这个字典可以用 DictConfig 对象来包装，以提供更好的配置管理。以下是对这段代码的逐行解释：
    1. def read_config(file: str) -> DictConfig:：这是函数的定义，它接受一个名为 file 的参数，该参数应该是一个字符串，
        表示 YAML 配置文件的路径。函数的返回类型被注释为 DictConfig，这暗示了返回值是一个包含配置的字典。
    2. if not os.path.exists(file):：此行代码检查指定的文件是否存在，如果文件不存在，则引发 FileNotFoundError 异常，并指示找不到文件。
    3. with open(file, "r", encoding="utf-8") as f:：这是一个上下文管理器，用于打开文件 file 以供读取。它以只读模式打开文件，并指定编码为 UTF-8。
    4. config = yaml.load(f, Loader=_PrettySafeLoader)：使用 PyYAML 库来加载 YAML 文件。
        yaml.load() 函数从文件 f 中读取 YAML 数据并将其解析为 Python 字典。
        Loader=_PrettySafeLoader 指定了 YAML 解析器的加载器，通常用于安全地加载 YAML 数据以防止潜在的安全漏洞。
    5. return OmegaConf.create(config)：返回一个包含配置信息的 DictConfig 对象。
        OmegaConf.create(config) 函数用于将 Python 字典转换为 DictConfig 对象，这使得配置更容易访问和管理。
总结，这个函数的主要作用是读取一个指定的 YAML 配置文件，将其解析为 Python 字典，然后将该字典包装为 DictConfig 对象，
以便在代码中更轻松地访问和管理配置信息。如果文件不存在或在读取和解析过程中出现错误，函数会引发相应的异常以进行错误处理。
"""
def read_config(file: str) -> DictConfig:
    """
    Reads a yaml configuration file.

    Args:
        file:  str, path of the yaml configuration file.

    Returns:
        dict (DictConfig), imported yaml file.
    """

    # Load configuration from file
    if not os.path.exists(file):
        raise FileNotFoundError(f'ERROR: Cannot find the file {file}\n')
    try:
        with open(file, "r", encoding="utf-8") as f:
            config = yaml.load(f, Loader=_PrettySafeLoader)
    except yaml.YAMLError as e:
        raise RuntimeError(f'ERROR: Cannot load the file {file}\n') from e

    return OmegaConf.create(config)


def write_config(data: DictConfig | prodict.Prodict, outfile: str) -> None:
    """
    Writes the dictionary data to a yaml file.

    Args:
        data:     dict (DictConfig), data to be stored as a yaml file.
        outfile:  str, path of the output file.
    """

    with open(outfile, "w", encoding="utf-8") as f:
        if isinstance(data, Prodict):
            yaml.dump(data.to_dict(is_recursive=True), f, indent=4, default_flow_style=None, sort_keys=False,
                      allow_unicode=True)
        elif isinstance(data, DictConfig):
            OmegaConf.save(config=data, f=f)
        else:
            yaml.dump(data, f, indent=4, default_flow_style=None, sort_keys=False, allow_unicode=True)
