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

Primary processing of MoTIF from its raw data occurs in [all_in_one_motif_preprocess.py](https://github.com/aburns4/MoTIF/blob/main/task_automation/seq2act/motif_data_generation/all_in_one_motif_preprocess.py). There are some quirks we'd like to bring your attention to, as you may want to make different design choices and update this code for your own purposes. Note that the raw data stored all view hierarchies as `jpg` extensions which was a human error made along the way, they are actually `json` files. Eventually we'll run a script to fix that and can upload the fixed version.

1. Matching UI Screen Interactions to UI View Hierarchy Elements
   - Finding the closest match
      The `gesture_to_ui` method takes a human screen interaction, which occurs on the visual rendering of the app screen, and computes the view hierarchy element which is closest to it. This requires the UI elements to be scaled to the screen dimensions for them to be in the same dimension space, and then we can compute which UI object is closest. Human interactions are captured as points on the screen and UI objects have bounding boxes. So, we took the midpoint of the latter and computed the Euclidean distance of between the interaction coordinate and the bbox midpoints.
   - Widget Exceptions
      Widgets pose a challenge for converting between the rendered app screen and the app view hierarchy in the backend. Not all widget view hierarchy jsons have a root dimension starting from [0, 0], because they contain the dimensions for the smaller widget part of the screen that's relevant for that time step. However, we rely on these left-corner root dimensions to define the scaling relationship between the rendered screen and backend. To try to fix this edge case, we take the correct root dimensions from other action sequence time steps of the same sample or other samples from that particular app. There are some samples that still weren't fixed by this, and as a result weren't used.
2. Unique Identifier (UID) per Screen
   - The `get_uid` class method creates an identifier per screen which is the concatenation of a majority of the text and attribute data stored in the UI object. This may be an imperfect way to define a unique identifier, and it doesn't compare visual similar or the fraction of pixels that are the same to consider a UI different from another.
3. Definition of Swiping Events
   - In the raw gesture data captured from user interaction with the app screens, the interaction is either stored as a single point (click) or a list of points (swipe). However, there are some edge cases in which someone slightly held down longer than is typical for a click and it results in a list of points with very little distance between them. In this case, these should not be considered swiping. For this reason, we only consider a list of points a swiping event in the `gesture_to_ui` method is the start to end swipe distance is larger than 0.01, which was defined by manually investigating the threshold. 
4. Definition of Typing Events
   Identifying typing events requires postprocessing of the action sequences. This is because there were two ways our collection interface could capture typing: from humans clicking on the phone's keyboard through our web interface or from humans typing from their computer keyboard. The latter is not directly captured during our collection and makes up a majority of the typing cases (since it is physically easier to do). Resultingly, we have to inspect the typable objects to see if they were typed into. 

   If a typable UI object has text in it and wasn't typed in already by a user interaction, we add it as a type event. This may not perfectly capture all edge cases in which some typable objects by default have filler text and are never interacted with. This processing can be improved in future work.
5. Deduplification of Action Sequences
   - Cycles
      Due to human annotators not always knowing how to complete an app task on the first attempt, the raw action sequence sometimes contains cyclic behavior. E.g., returning to the home screen. We provide an input argument to turn on or off the deduplification of these cycles. We always remove duplicate states if they are back to back and have the same action and action location, since this may just be due to collection interface delays. See the `clean_idxs` function for this processing.
      
      For cycles, we essentially "start over" the action sequence from the finished cycle. This is a challenging case to programmatically resolve- this processing may not perfectly dedupe all samples, or may dedupe more often than necessary. We invite others to try different ways to do this, feel free to open an issue on the repo with your ideas.
   - Sequential Typing
      We use the `get_text_duplicates` function to find typing actions that all constitute the same type action. This might happen if our data collection interface captures multiple moments of a typing event or if a user individually typed each letter on the rendered phone keyboard, they might appear as independent time steps when they are actually all associated with the same action. For example, someone could incrementally type "p" "pi" "pin" "pink" which are all for the final type event "pink".
6. Extra Cleaning
   - Logging in Events
   - Technical Errors (Google Play Store keeps stopping)
   - Missing Gesture
   - JSON Error
   - End State
7. Step by Step Instructions

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