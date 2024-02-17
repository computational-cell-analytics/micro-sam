def base_slurm_script(tier_choice, env_name, resource_name=None):
    if tier_choice == 1:
        assert resource_name in ["V100", "A100"]
        # here's where we train on A100 / V100 (cluster-based high-end GPUs)
        base_script = f"""#!/bin/bash
        #SBATCH -c 16
        #SBATCH --mem 64G
        #SBATCH -t 2-00:00:00
        #SBATCH -p grete:shared
        #SBATCH -G {resource_name}:1
        #SBATCH -A gzz0001"""

    elif tier_choice == 2:
        assert resource_name in ["v100", "gtx1080"]
        # here's where we train on GTX1080 (workstation level GPUs)
        base_script = f"""#!/bin/bash
        #SBATCH -p gpu
        #SBATCH --mem 64G
        #SBATCH -G {resource_name}:1
        #SBATCH -c 16
        #SBATCH -t 2-00:00:00"""

    elif tier_choice == 3:
        # run on CPU nodes (would test it on laptops, and on CPU nodes)
        base_script = """#!/bin/bash
        #SBATCH -p medium
        #SBATCH --mem 64G
        #SBATCH -c 16
        #SBATCH -t 2-00:00:00"""

    else:
        raise ValueError("The provided tier does not exist")
    
    base_script += "\n" + f"source activate {env_name}" + "\n"

    return base_script
