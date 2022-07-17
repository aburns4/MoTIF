# replace decode_folder below with the appropriate folder 
with open('motif/decode_folder/decodes.joint_act') as f:
    results = f.readlines()
    results = [x.strip() for x in results]

in_order = True

gt_pred_seq = []
for i in range(0, len(results), 4):
    g = 'gt_seq'
    p = 'pred_seq'
    if i == (len(results)-1):
        break
    gt = results[i+1][len(g):].replace('- ', '').split(' ')[:-1]
    pred = results[i+2][len(p):].replace('- ', '').split(' ')
    gt_pred_seq.append([gt, pred])

gt_pred_act = []
for res in gt_pred_seq:
    gt_act = [int(x.split(',')[1]) for x in res[0]]
    try:
        pred_act = [int(x.split(',')[1]) for x in res[1]]
    except:
        pred_act = []
    gt_pred_act.append([gt_act, pred_act])

################################
gt_pred_ref = []
for res in gt_pred_seq:
    gt_ref = [int(x.split(',')[0]) for x in res[0]]
    try:
        pred_ref = [int(x.split(',')[0]) for x in res[1]]
    except:
        pred_ref = []
    gt_pred_ref.append([gt_ref, pred_ref])


total_complete_action = []
total_partial_action = []
for sample in gt_pred_ref:
    complete_action = 1.0
    partial_action = 0.0
    seq_len = min(len(sample[0]), len(sample[1]))
    if seq_len == 0:
        complete_action = 0.0

    for i in range(seq_len):
        if sample[0][i] != sample[1][i]:
            complete_action = 0.0
            if in_order:
                break
        if sample[0][i] == sample[1][i]:
            partial_action += 1 / seq_len
    total_complete_action.append(complete_action)
    total_partial_action.append(partial_action)

################################
total_complete_grounding = []
total_partial_grounding = []
for sample in gt_pred_act:
    complete_grounding = 1.0
    partial_grounding = 0.0
    seq_len = min(len(sample[0]), len(sample[1]))
    if seq_len == 0:
        complete_grounding = 0.0

    for i in range(seq_len):
        if  (abs(sample[0][i] - sample[1][i]) > 1): # (sample[0][i] != sample[1][i]) 
            complete_grounding = 0.0
            if in_order:
                break
        if (abs(sample[0][i] - sample[1][i]) <= 1):
            partial_grounding += 1 / seq_len
    total_complete_grounding.append(complete_grounding)
    total_partial_grounding.append(partial_grounding)

#################################
gt_pred_ref_and_act = []
for res in gt_pred_seq:
    gt_act = [[int(x.split(',')[0]), int(x.split(',')[1])] for x in res[0]]
    try:
        pred_act = [[int(x.split(',')[0]), int(x.split(',')[1])] for x in res[1]]
    except:
        pred_act = []
    gt_pred_ref_and_act.append([gt_act, pred_act])

total_complete_action_grounding = []
total_partial_action_grounding = []
for sample in gt_pred_ref_and_act:
    complete_action_grounding = 1.0
    partial_action_grounding = 0.0
    seq_len = min(len(sample[0]), len(sample[1]))
    if seq_len == 0:
        complete_action_grounding = 0.0

    for i in range(seq_len):
        if (sample[0][i][0] != sample[1][i][0]) or (abs(sample[0][i][1] - sample[1][i][1]) > 1): # (sample[0][i][1] != sample[1][i][1]):
            complete_action_grounding = 0.0
            if in_order:
                break
        if (sample[0][i][0] == sample[1][i][0]) and (abs(sample[0][i][1] - sample[1][i][1]) <= 1): # (sample[0][i][1] == sample[1][i][1]):
            partial_action_grounding += 1 / seq_len
    total_complete_action_grounding.append(complete_action_grounding)
    total_partial_action_grounding.append(partial_action_grounding)

print('Length of tasks?? %d' % len(total_complete_action))
mean_comp_act_ground = sum(total_complete_action_grounding) / (len(total_complete_action_grounding))
mean_part_act_ground = sum(total_partial_action_grounding) / (len(total_partial_action_grounding))
print('Mean complete action class + grounding accuracy = %f' % mean_comp_act_ground)
print('Mean partial action class + grounding accuracy = %f' % mean_part_act_ground)

mean_comp_ground = sum(total_complete_grounding) / (len(total_complete_grounding))
mean_part_ground = sum(total_partial_grounding) / (len(total_partial_grounding))
print('Mean complete grounding accuracy = %f' % mean_comp_ground)
print('Mean partial grounding accuracy = %f' % mean_part_ground)

mean_comp_action = sum(total_complete_action) / (len(total_complete_action))
mean_part_action = sum(total_partial_action) / (len(total_partial_action))
print('Mean complete action accuracy = %f' % mean_comp_action)
print('Mean partial action accuracy = %f' % mean_part_action)
