#%%
import pandas as pd 
import wandb
api = wandb.Api()

def pop_tolist(df, column: str):
    return df.join(pd.DataFrame(df.pop(column).tolist())) 
# Project is specified by <entity/project-name>
runs = api.runs("data-intelligence-challenge-g16/data-intelligence-challenge")

summary_list, config_list, name_list = [], [], []
for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append(
        {k: v for k,v in run.config.items()
          if not k.startswith('_')})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame({
    "summary": summary_list,
    "config": config_list,
    "name": name_list
    }).pipe(pop_tolist, column="summary").pipe(pop_tolist, column="config")
runs_df

#%%
runs_df.to_csv("project.csv")