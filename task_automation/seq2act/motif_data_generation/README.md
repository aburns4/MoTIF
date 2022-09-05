# MoTIF Data Processing

We provide instructions for running the data processing files in the above folder (Seq2Act/README.md). Here, we provide additional information on what the processing code is doing, certain design choices that were made, and the data format and fields.

If you use our data, please cite our paper:
```
@inproceedings{burns2022motifvln,
      title={A Dataset for Interactive Vision Language Navigation with Unknown Command Feasibility}, 
      author={Andrea Burns and Deniz Arsan and Sanjna Agrawal and Ranjitha Kumar and Kate Saenko and Bryan A. Plummer},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
}
```

## Design Choices and Random Notes

Primary processing of MoTIF from its raw data occurs in [all_in_one_motif_preprocess.py](https://github.com/aburns4/MoTIF/blob/main/task_automation/seq2act/motif_data_generation/all_in_one_motif_preprocess.py). There are some quirks we'd like to bring your attention to, as you may want to make different design choices and update this code for your own purposes.

1. Matching UI Screen Interactions to UI View Hierarchy Elements
   - Finding the closest match
   - Widget Exceptions
2. Unique Identifier (UID) per Screen
3. Definition of Swiping Events
4. Definition of Typing Events
5. Deduplification of Action Sequences
6. Extra Cleaning
   - Logging in Events
   - Technical Errors (Google Play Store keeps stopping)

## JSON Data Format
The JSON data we refer to here is the result of running dedup.sh on the raw data folders. You can directly download the processed json data and skip this processing step. Here is what is stored in each JSON file:

| Feature Name            | Type         | Content                                  |
| ----------------------- | ------------ | ---------------------------------------- |
| trace_id                |  str         | A unique id for each action sequence     |
| goal                    |  str         | Natural language sentence describing a high-level instruction for a request on an Android app |
| instr                   | list(str)    | Synthetic sentences describe the low-level (step-by-step) instruction for a request on an Android app |
| app                     | str          | Android app package name                 |
| screen_w                | int          | Width of the rendered Android screen     | 
| screen_h                | int          | Height of the rendered Android screen    |
| vh_w                    | int          | Width with respect to the view hierarchy |
| vh_h                    | int          | Height with respect to the view hierarchy|
| scale_x                 | float        | Width relationship between the view hierarchy and screen rendering coordinate system |
| scale_y                 | float        | Height relationship between the view hierarchy and screen rendering coordinate system |
| actions                 | list(str)    | List of action class at each time step, action is one of (click, type, swipe) |
| verb_str                | list(str)    | List of the verb string used to populate the synthetic step-by-step instructions |
| obj_desc_str            | list(str)    | List of the object description string used to populate the synthetic step-by-step instructions |
| input_str               | list(str)    | List of the typing input string for step-by-step instructions (empty if the corresponding time step does not have a type action)|
| ui_types                | list(int)    | List of the type of element interacted with for each action, defined by the UIObjectType Enum (see view_hierarchy.py) |
| screen_bboxes           | list(float)  | List of [x1, y1, x2, y2] bounding boxes for each UI element interacted with in the action sequence; coordinates with respect to the rendered screen   |
| view_bboxes             | list(int)    | List of [x1, y1, x2, y2] bounding boxes for each UI element interacted with in the action sequence; coordinates are from the view hierarchy bounding boxes |
| images                  | list(str)    | List of unique image (also view hierarchy) files that reflect action sequence |
| raw_gestures            | list(float)  | List of raw [x, y] gestures captured on our interface. These are normalized [0, 1] with respect to the screen. If a time step has a list of more than one [x, y] point, it is a swipe event. |
| ui_target_idxs          | list(int)    | List of indices that identify which UI element the user action action occured on; this is the index of the UI element among the leaf node list at that time step |


## TFRecord Data Format
These features are the kept the same as the original Seq2Act formatting for RicoSCA and PixelHelp.

| Feature Name            | Type   | Content                                  |
| ----------------------- | ------ | ---------------------------------------- |
| Task_id                 | int    | an unique id for each example            |
| Instruction_str         | string | English sentence describes a list of actions users can do sequentially on Android |
| Instruction_word_id_seq | int    | split Instruction_str to words and give each word an unique id |
| Instruction_rule_id     | int    | if the sentence is real or synthetic     |
| Ui_obj_str_seq          | string | English phrases/text of UI elements in all screens | 
| Ui_obj_word_id_seq      | int    | split Ui_obj_str_seq to words and give each word an unique id |
| Ui_obj_type_id_seq      | int    | UI element type                          |
| Ui_obj_clickable_seq    | int    | if UI element is clickable               |
| Ui_obj_cord_x_seq       | int    | X coordinates of UI elements             |
| Ui_obj_cord_y_seq       | int    | Y coordinates of UI elements             |
| Ui_obj_v_distance       | int    | vertical distance of any two UI elements |
| Ui_obj_h_distance       | int    | horizontal distance of any two UI elements |
| Ui_obj_dom_distance     | int    | distance of any two UI elements in dom tree |
| Ui_obj_dom_location_seq | int    | traversal order of a UI element in the dom tree |
| Verb_id_seq             | int    | verb for the action |
| Ui_target_id_seq        | int    | which UI elements is user actioned on    |
| Input_str_position_seq  | int    | the position of the string user input in the Instruction_str |
| Obj_desc_position_seq   | int    | the position of the substring of the UI element in the Instruction_str. |