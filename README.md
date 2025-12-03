Here is my workflow:
1. Create a Wandb sweep (and project if it doesn't exist). Define the search space in config.search_space. Define the sweep name and number of neural network layers
 in config.wandb_sweep. Define the values in config.wandb. Run
    `python  init_wandb_sweep.py`

     The script will print the sweep_id. Copy the script_id into config.wandb_sweep.sweep_id.

2. In addition to the config values defined above, define the first the values from "mode" to "optimizer" in the config file. Set config.wandb_sweep.count to the number of trials
   you want executed in a single job. To launch a hyperparameter sweep, run
   `python run.py`

3. Downloading trained models. Edit download_folder and artifact_name in artifact_download.py. You can find the artifact_name by clicking on artifact section of a wandb run.
   Download the artifact by running:
   `python download_artifact.py`

Using CHTC. Heres an intro <https://chtc.cs.wisc.edu/uw-research-computing/htc-roadmap#main>.
1. Place the CHTC submit files (build.sub, sweep.sub, sweep.sh) outside of the tdgoal repository.
2. Build the .sif image for the container you will be using. I already wrote the build.sub file. Follow the guide at <https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc#build-your-own-container>
   to build the image.
3. Add your wandb_api in sweep.sh. Edit sweep.sub to change the resources you want to request per job and the number of jobs you want to submit (the number after the word "queue").
   If each run is short, then put all the runs in one job: config.wandb_sweep.count=<num_trials> and submit only one job. If each run is long, then set config.wandb_sweep.count=1 and
   and edit total number of trials you want as number of jobs in sweep.sub. Then it will launch all of the runs as seperate jobs running in parallel. To the submit the job(s), run

   `condor_submit sweep.sub`
