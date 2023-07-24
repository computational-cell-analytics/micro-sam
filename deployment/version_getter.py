import runpy
import ruamel.yaml
import os

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
ctor_fname = os.path.join("construct.yaml")

with open(ctor_fname, 'r') as stream:
    ctor_conf = yaml.load(stream)

ctor_conf["version"] = runpy.run_path(os.path.join("..", "micro_sam", "__version__.py"))["__version__"]

with open(ctor_fname, 'w') as outfile:
    yaml.dump(ctor_conf, outfile)
