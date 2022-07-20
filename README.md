# A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility
This repository contains the code for the data processing, models, and experimental framework for "A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility" Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, Bryan A. Plummer, which is accepted in the proceedings of the European Conference on Computer Vision (ECCV) 2022.

## MoTIF Dataset
If you want the most general form of the interaction data, you can access the aggregated data here! This contains the rendered app screens, app view hierarchies, raw interaction sequences, processed jsons containing cleaned action sequences (with action classes, both high-level and low-level instruction, element bounding boxes, etc.), and task feasibility annotations.

Note that each trace subdirectory contains a `task_type.txt` text file. There are three things that can appear here: "DEMONSTRATION", "EXPLORE EVERYTHING", or "LANGUAGE ANNOTATION." The exploration traces were collected in the process of trying to generate state-action space graphs, they capture more unique states but do not have paired language data. The demonstration and language annotation folders have a `task.txt` text file which contains the demonstration's natural language instruction or all written tasks for the app during language collection.

All of the language annotations and the data captured during that stage of MoTIF's collection is under the `language collection` directory. If you are interested in generating the specific data files and features for our task feasibility or task automation experiments, please see the `task_feasibility` and `task_automation` directories, respectively.

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
