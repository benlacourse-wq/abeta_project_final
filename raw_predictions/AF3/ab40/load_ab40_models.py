from pymol import cmd
import os
import glob

base_dir = r"C:\Users\benla\OneDrive\Documents\Bioinformatics_MSc\LIFE703\Data\AF3\ab40"

for oligomer in range(1, 16):
    folder = os.path.join(base_dir, f"ab40_{oligomer}mer_templates")

    for model in range(5):
        filename = os.path.join(
            folder,
            f"fold_ab40_{oligomer}mer_templates_model_{model}.cif"
        )

        if os.path.exists(filename):
            obj_name = f"ab40_{oligomer}mer_m{model}"
            cmd.load(filename, obj_name)
            print(f"Loaded {obj_name}")