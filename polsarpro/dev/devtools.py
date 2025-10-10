from collections import OrderedDict
import re

def parse_psp_parameter_string(input_string):
    """Convenience development function to make cli parameters from strings copied from parameter files."""
    ordered_dict = OrderedDict()
    
    for line in input_string.strip().split("\n"):
        key, value = re.split(r':\s*', line, maxsplit=1)
        if value.isdigit():
            value = int(value)
        elif value.replace('.', '', 1).isdigit():  # Check for float
            value = float(value)
        ordered_dict[key] = value
    str_param = " ".join([f"-{it[0]} {it[1]}" for it in ordered_dict.items()])
    return str_param