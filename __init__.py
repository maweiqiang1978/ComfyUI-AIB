from pathlib import Path
import sys
import os
import importlib.util


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

current_dir = Path(__file__).parent
nodes_folder = current_dir / 'nodes_folder'

for filename in os.listdir(nodes_folder):
	if filename.startswith('_'):
		continue
	module_name = filename[: -3]
	module_path = os.path.join(nodes_folder, filename)
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(module)
	NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
	NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]