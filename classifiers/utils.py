import torch
import inspect
import numpy as np
from typing import Dict, Optional, Any, Type

def replace_module(module: torch.nn.Module, _class_: Optional[Type] = None, **kwargs: Any) -> torch.nn.Module:
    """
    Replaces `module` with a new module, with different constructor arguments and possibly of a new class type.
    Arguments not specified for override will be preserved from the original module. If `_class_` is specified,
    only arguments that are part of `_class_`'s constructor will be preserved.

    Examples:
    ```python
    # replacing one module attribute and preserving all others
    mod = replace_module(torch.nn.Conv2d(..., out_channels=2, kernel_size=3), kernel_size=4)
    returns torch.nn.Conv2d(..., out_channels=2, kernel_size=4)
    ```

    ```python
    # Changing the module class and preserving all attributes
    mod = replace_module(torch.nn.Conv2d(..., kernel_size=3), _class_=torch.nn.LazyConv2d)
    returns torch.nn.LazyConv2d(..., kernel_size=3)
    ```

    Parameters
    ----------
    module : torch.nn.Module
        The module to replace.
    _class_ : Optional[Type], optional
        The desired class of the new module, by default None
    **kwargs : Any
        The argument overrides to supply to the constructor of the new module

    Returns
    -------
    torch.nn.Module
        The new module.
    """
    _class_ = _class_ or module.__class__
    class_params = inspect.signature(_class_).parameters.keys()
    module_params = {p: getattr(module, p) for p in class_params if hasattr(module, p)}
    if "bias" in module_params and isinstance(module_params["bias"], torch.nn.Parameter):
        module_params["bias"] = True
    module_params.update(kwargs)
    new_module = _class_(**module_params)
    return new_module


def replace_submodules(module: torch.nn.Module, **replacements: Dict[str, Any]) -> torch.nn.Module:
    """
    Replaces submodules by name. Tries to preserve all attributes from the original module that
    aren't explicitly overridden.

    Parameters
    ----------
    module : torch.nn.Module
        The parent module on which to modify submodules.
    **replacements : Dict[str, Any]
        The keys should be string submodule names, joined by `"_"`. Example: "features_conv0"
        The values should be dicts of arguments to `photosynthetic.training.nn.module_utils.replace_module`.
        See that docstring for more information.

    Returns
    -------
    torch.nn.Module
        The parent module with modified submodules.
    """
    for target, replacement_kwargs in replacements.items():
        parent_name, _, child_name = target.rpartition("_")
        parent = module.get_submodule(parent_name.replace("_", "."))
        child = parent.get_submodule(child_name)
        new_module = replace_module(module=child, **replacement_kwargs)
        setattr(parent, child_name, new_module)
    return module

def compute_class_weights(y: np.ndarray) -> np.ndarray:
    assert len(y.shape) == 1,f"Label should be 1-d but got {y.shape}"
    _, counts = np.unique(y, return_counts = True)
    class_probabilities = counts/len(y)
    n_classes = len(class_probabilities)
    return n_classes, (1 / (n_classes * class_probabilities))
