import importlib
from os import path as osp
from pdb import set_trace as stx
from basicsr.utils import scandir

arch_folder = osp.dirname(osp.abspath(__file__))
arch_filenames = [
    osp.splitext(osp.basename(v))[0]
    for v in scandir(arch_folder)
    if v.endswith("_arch.py")
]
_arch_modules = [
    importlib.import_module(f"basicsr.models.archs.{file_name}")
    for file_name in arch_filenames
]

def dynamic_instantiation(modules, cls_type, opt):
    for module in modules:
        cls_ = getattr(module, cls_type, None)
        if cls_ is not None:
            break
    if cls_ is None:
        raise ValueError(f"{cls_type} is not found.")
    return cls_(**opt)
def define_network(opt):
    network_type = opt.pop("type")
    net = dynamic_instantiation(_arch_modules, network_type, opt)
    return net
