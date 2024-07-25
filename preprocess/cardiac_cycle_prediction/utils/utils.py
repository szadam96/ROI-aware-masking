import importlib

def get_name(obj):
    name = str(obj)
    if name.endswith('()'):
        name = name[:-2]
    name = name.lower()

    return name


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported class: {class_name}')
