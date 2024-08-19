# A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility
This repository contains the code for the data processing, models, and experimental framework for "A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility" Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, Bryan A. Plummer, which is accepted in the proceedings of the European Conference on Computer Vision (ECCV) 2022.

## MoTIF Dataset
If you want the most general form of the interaction data, you can access the aggregated data [here](https://drive.google.com/file/d/1Zxm-onsO5MURcKYrRVqJjov0Zb3U1lGf/view?usp=sharing)! This contains the rendered app screens, app view hierarchies, raw interaction sequences, language annotations, and task feasibility annotations. Additional processing to clean the raw data to a processed json format can be found in the task automation folder [here](https://github.com/aburns4/MoTIF/blob/main/task_automation/seq2act/motif_data_generation/all_in_one_motif_preprocess.py).

<img src="https://github.com/aburns4/MoTIF/blob/main/motif.jpg" alt="Graphic illustrating feasible and infeasible MoTIF mobile app action sequences" width="825">
 
Note that each trace subdirectory contains a `screens` folder with app screenshots, `view_hierarchies` folder with android view hierarchies, a `metadata.json` file containing user interactions, and a `trace_info.json` file.

This `trace_info.json` file contains four types of traces, which is defined in the `trace_type` key:
1. _language_: these are traces collected as a _byproduct_ of obtaining annotations for written commands users would want to automate in the apps. These jsons also have an `all_tasks` key which contains a string of comma separated commands, as written by our human annotators for the given app. Note that the comma separation was not perfect in separating commands, and could use further human cleaning potentially. We only collected action traces / task demonstration for a subset of these. These folders only have screens and not view hierarchies because at the time they were not needed and the latter take up a lot of storage space.
2. _action_: these traces are human attempts at completing a user specified instruction in the provided app. They are used for task automation. They also have keys `seconds_spent` which is the time an annotator spent interacting with the app, `task_command` the associated high level goal the action trace reflects, and `feasibility`, a dictionary with `binary`, `reason`, and `follow_up` keys reflecting the binary feasibility label, the subclass reason for why if `binary` is No, and the free form follow up questions annotators wrote to either clarify/correct the original task input or see if the user wanted to perform a different task.
3. _exploration_: these traces are human attempts at exploring as much of the provided app as possible. At first we tried to collect these traces for building an offline state-action graph for each app but found it introduced a bit of noise, so we didn't use them. They capture more unique states but do not have paired language data.
4. _other_: Other traces which came from any of the above but had missing metadata or other technical glitches deeming them unusable for train/test data. Still, the screens or view hierarchy data may be available and of interest for other uses. 

If you are interested in generating the specific data files and features for our task feasibility or task automation experiments, please see the `task_feasibility` and `task_automation` directories, respectively.

### Citation
If you use our code or data, please consider citing our work :)
```
@inproceedings{burns2022motifvln,
      title={A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility}, 
      author={Andrea Burns and Deniz Arsan and Sanjna Agrawal and Ranjitha Kumar and Kate Saenko and Bryan A. Plummer},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
}
```
