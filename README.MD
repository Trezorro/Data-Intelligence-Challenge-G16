# Data Intelligence Challenge solution

https://user-images.githubusercontent.com/19216539/178445143-edc68b35-0e06-4cb1-992a-064c88522db8.mp4

_Demo of our continuous simulation environment with a PPO-based agent, trained for 300 epochs_

## Continuous Simulation Setup
We recommend using (mini)conda for installing the dependencies in a separate new environment. This makes installing Pytorch on CUDA and other libraries a little easier. Installing dependencies may be done using `pip install -r requirements.txt`, but we provide no guarantees for this method.


### Steps:
1. Install MPI
   - For Windows:
     1. Go to [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=100593).
     2. Click the Download button and select only the setup file `msmpisetup.exe`.
     3. Run the downloaded file and install Microsoft MPI.
     4. Confirm that Microsoft MPI has been added to your path.
        1. Either click Start, type Run, and press Enter OR press Windows+R.


        2. Type `sysdm.cpl` and click "OK".
        3. Click the "Advanced" tab and then click "Environment Variables...".
        4. Under the "System Variables" panel (lower square), find the variable "Path" and confirm that `C:\Program Files\Microsoft MPI\Bin` (or wherever you installed Microsoft MPI) is there.
           1. If yes, press "OK" to close the panel then "OK" again to close the System Properties panel.
           2. If not, click the "Path" variable, click "Edit", and add `${Microsoft MPI Install Location}\Bin`, replacing `${Microsoft MPI Install Location}` with wherever you installed Microsoft MPI 
   - For Ubuntu: get OpenMPI `sudo apt-get update && sudo apt-get install libopenmpi-dev`
   - For Max OSX: get OpenMPI `brew install openmpi`
   - On some system configurations, you can install OpenMPI using conda `conda install openmpi`. Not guaranteed to work.
2. Replicate the conda environment using:
   1. `conda env create -f conda_env.yaml`
   2. `conda activate dic-env`
3. Install the spinup module and it's dependencies using: `pip install -e spinningup` (inside your conda environment.)
4. Running training and evaluation:
   - To train the vacuum robot: `python .\Continuous-Simulations\runner.py`
   - Watch trained robot policy by running `python .\Continuous-Simulations\watch_trained_bot.py`. A ready to go algorithm is already included in this folder.
   - For Windows: if there is a Windows Firewall popup, allow Python to communicate through the firewall.

**Note** that the C++ build tools on your system are required for using this project.

**Note (2)** that your path variable will now have the `<path>\swig-<version>` and `<path>\Microsoft MPI\bin`. 
On windows, your path variable can be set by searching for `edit the system environment variables` -> `environment variables` -> Find `Path` variable -> `Edit` -> Add the paths.

