import runpy
import ruamel.yaml
import os

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

# Get the current OS name from the environment variable
current_os = os.environ.get('RUNNER_OS').lower()  # Ensure lowercase for comparison

# Construct the filename based on OS name (consider all possibilities)
if current_os == 'windows':
    construct_file = os.path.join("construct_windows-latest.yaml")
elif current_os == 'linux':
    construct_file = os.path.join("construct_ubuntu-latest.yaml")  # Assuming ubuntu-latest for Linux
# Add an else block if you plan to support macOS in the future
else:
    raise Exception(f"Unsupported OS: {current_os}")

# Load YAML using ruamel.yaml
yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True

with open(construct_file, "r") as stream:
    ctor_conf = yaml.load(stream)

ctor_conf["version"] = runpy.run_path(os.path.join("..", "micro_sam", "__version__.py"))["__version__"]

with open(construct_file, "w") as outfile:
    yaml.dump(ctor_conf, outfile)