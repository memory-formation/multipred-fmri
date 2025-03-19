# Instructions

This repository contains the scripts and code required to reproduce the analyses for the study **[]**.

## Environment Setup
1. Clone this repository:

   ```bash
   git clone https://github.com/memory-formation/multipred-fmri.git
   ```

2. To ensure compatibility, use the provided `pyproject.toml` :

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -e . # This command will install requirements listed in the pyproject.toml
```

## Dataset
The dataset is hosted on OpenNeuro and **must be downloaded separately**. To ensure proper integration:
1. Download the dataset from OpenNeuro:
   - [Dataset Link](https://openneuro.org/datasets/XXXXX)
2. Place the dataset in the cloned repository folder, and rename it as *data/*:
   ```
   mulitpred-fmri/
   ├── data/  # Place the downloaded dataset here
   ├── analyses/
   ├── multipred/
   ├── pyproject.toml
   ├── README.md
   ```
## Dataset Structure
The dataset follows BIDS conventions. Raw data (functional and structural (defaced) T1)

```
mulitpred-fmri/
├── data/                # Place OpenNeuro dataset here
    ├── sub-XX/
    │   ├── anat/ # Structural defaced T1
    │   ├── func/ # functional data from each run
    ├── derivatives/             # Processed outputs
    │   ├── behavioral_raw/      # raw behavioral data, and preprocessed datasets of the whole sample
    │   ├── fsl_firstlevel/      # First-level FSL results
    │   ├── fsl_preprocessed/     # Second-level
    │   ├── fsl_secondlevel/     # Second-level 
    │   ├── fsl_thirdlevel/      # Third-level
    │   ├── masks/              # different ROI masks of each subject
    │   ├── single_trial_estimates/   # single trial copes (FSL, least squares separate)
    │   ├── skullstripped_T1/    # skullstripped structural volumes (using BET)
    ├── code/              # Python and bash scripts used to obtain (almost) every derivative file. 
                            #Just make sure to update the path at the start of each script, and to have an installation of FSL 6.0.0
    │   ├── behav_preproc/
    │   ├── experiment/
    │   ├── fsl_feat_scripts/
    │   ├── FSL_MNI_templates/
    │   ├── func_localizer_voxels_masks/
    │   ├── single_trial_estimates/
    │   ├── SVC_decoding/
```

## Running Analyses
Once you complete the setup and downloaded the dataset you will be able to run the notebooks that can be found in analyses.
Each notebook contains the code to run all analyses and save figures (in analyses/figures/)

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
