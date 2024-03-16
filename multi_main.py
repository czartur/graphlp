import subprocess
import itertools
import argparse
import os
from dataclasses import dataclass, fields, field
from typing import List, get_origin

@dataclass
class Configuration:
    training_size: List[str] = field(default_factory=lambda: ["10"])
    model: List[str] = field(default_factory=lambda: ["NLP"])
    nlp_initial_embedding_from: List[str] = field(default_factory=lambda: ["DDP", "ISO"])
    ddp_projection: List[str] = field(default_factory=lambda: ["pca", "barvinok"])
    similarity: List[str] = field(default_factory=lambda: ["path", "wup"])
    output_path: str = "trained"

def dataclass_to_argparse(dc):
    parser = argparse.ArgumentParser()
    for dc_field in fields(dc):
        field_type = dc_field.type
        field_name = dc_field.name.replace('_', '-')
        if get_origin(field_type) is list:
            parser.add_argument(
                f'--{field_name}',
                nargs='+',
                default=dc_field.default_factory(),
                help=f'{field_name} (default: {dc_field.default_factory()})'
            )
        else:
            parser.add_argument(
                f'--{field_name}',
                default=dc_field.default,
                help=f'{field_name} (default: {dc_field.default})'
            )
    return parser

def parse_args_to_dataclass(dc_cls):
    parser = dataclass_to_argparse(dc_cls)
    args = parser.parse_args()
    return dc_cls(**vars(args))

def main():
    config = parse_args_to_dataclass(Configuration)
    
    output_path = os.path.join(os.getcwd(), config.output_path)
    os.makedirs(output_path, exist_ok=True)

    config_dict = vars(config)
    del config_dict["output_path"]
    combinations = list(itertools.product(*config_dict.values()))
    
    print(combinations)
    for i, combination in enumerate(combinations, start=1):
        
        # add path, lil dirty
        cmd_path_suffix = '_'.join(combination) + ".pkl"
        combination = combination + (os.path.join(output_path, cmd_path_suffix),)
        config_dict["save_path"] = ""

        cmd_args = [] 
        for field_name, field_value in zip(config_dict.keys(), combination):
            cmd_args.append(f"--{field_name.replace('_', '-')}={field_value}")
        cmd = ["python3", "main.py", "--no-plot"] + cmd_args
    
        print()
        print(f"#RUN {i}/{len(list(combinations))}:")
        print(' '.join(cmd))
        print()

        subprocess.run(cmd)

if __name__ == "__main__":
    main()

