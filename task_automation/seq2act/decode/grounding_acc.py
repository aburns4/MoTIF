def get_decode_results(filename):
    # load file where decoded outputs are stored
    print('----------' + filename.split('/')[-2] + '----------')
    with open(filename) as f:
        results = f.readlines()
        results = [x.strip() for x in results]
    return results

def get_gt_and_pred(results):
    gt_pred_seq = []
    for i in range(0, len(results), 4):
        g = 'gt_seq'
        p = 'pred_seq'
        if i == (len(results)-1):
            break
        gt = results[i+1][len(g):].replace('- ', '').split(' ')[:-1]
        pred = results[i+2][len(p):].replace('- ', '').split(' ')
        gt_pred_seq.append([gt, pred])
    return gt_pred_seq

def get_action_gt_and_pred(gt_pred_seq):
    gt_pred_act = []
    for res in gt_pred_seq:
        gt_act = [int(x.split(',')[0]) for x in res[0]]
        try:
            pred_act = [int(x.split(',')[0]) for x in res[1]]
        except:
            pred_act = []
        gt_pred_act.append([gt_act, pred_act])
    return gt_pred_act

def get_ground_gt_and_pred(gt_pred_seq):
    gt_pred_ref = []
    for res in gt_pred_seq:
        gt_ref = [int(x.split(',')[1]) for x in res[0]]
        try:
            pred_ref = [int(x.split(',')[1]) for x in res[1]]
        except:
            pred_ref = []
        gt_pred_ref.append([gt_ref, pred_ref])
    return gt_pred_ref

def get_action_accuracy(gt_pred, complete=True):
    total_action = []
    if complete:
        for sample in gt_pred:
            complete_action = 1.0
            if len(sample[1]) == 0 or (len(sample[0]) != len(sample[1])):
                complete_action = 0.0
                total_action.append(complete_action)
                continue

            for i in range(len(sample[0])):
                if sample[0][i] != sample[1][i]:
                    complete_action = 0.0
                    break
            total_action.append(complete_action)
    else:
        # partial accuracy
        for sample in gt_pred:
            partial_action = []
            seq_len = min(len(sample[0]), len(sample[1]))
            for i in range(seq_len):
                if sample[0][i] == sample[1][i]:
                    partial_action.append(1.0) 
            score = sum(partial_action) / len(sample[0])
            total_action.append(score)
    return sum(total_action) / len(total_action)

def get_ground_accuracy(gt_pred, complete=True):
    total_grounding = []
    if complete:
        for sample in gt_pred:
            complete_grounding = 1.0
            if len(sample[1]) == 0 or (len(sample[0]) != len(sample[1])):
                complete_grounding = 0.0
                total_grounding.append(complete_grounding)
                continue

            for i in range(len(sample[0])):
                if (abs(sample[0][i] - sample[1][i]) > 1):
                # if (sample[0][i] != sample[1][i]): # exact match code 
                    complete_grounding = 0.0
                    break
            total_grounding.append(complete_grounding)
    else:
        # partial accuracy
        for sample in gt_pred:
            partial_ground = []
            seq_len = min(len(sample[0]), len(sample[1]))
            for i in range(seq_len):
                if (abs(sample[0][i] - sample[1][i]) <= 1): 
                # if (abs(sample[0][i] - sample[1][i]) == 0): # exact match code
                    partial_ground.append(1.0) 
            score = sum(partial_ground) / len(sample[0])
            total_grounding.append(score)
    return sum(total_grounding) / len(total_grounding)

def get_both_gt_and_pred(gt_pred_seq):
    gt_pred_both = []
    for res in gt_pred_seq:
        gt_act = [[int(x.split(',')[0]), int(x.split(',')[1])] for x in res[0]]
        try:
            pred_act = [[int(x.split(',')[0]), int(x.split(',')[1])] for x in res[1]]
        except:
            pred_act = []
        gt_pred_both.append([gt_act, pred_act])
    return gt_pred_both

def get_both_accuracy(gt_pred, complete=True):
    total_both = []
    if complete:
        for sample in gt_pred:
            complete_both = 1.0
            seq_len = min(len(sample[0]), len(sample[1]))
            if len(sample[1]) == 0 or len(sample[0]) != len(sample[1]):
                complete_both = 0.0
                total_both.append(complete_both)
                continue

            for i in range(seq_len):
                if (sample[0][i][0] != sample[1][i][0]) or (abs(sample[0][i][1] - sample[1][i][1]) > 1):
                # if (sample[0][i][0] != sample[1][i][0]) or (sample[0][i][1] != sample[1][i][1]): # exact match code
                    complete_both = 0.0
                    break
            total_both.append(complete_both)
    else:
        # partial accuracy
        for sample in gt_pred:
            partial_both = []
            seq_len = min(len(sample[0]), len(sample[1]))
            for i in range(seq_len):
                if (sample[0][i][0] == sample[1][i][0]) and (abs(sample[0][i][1] - sample[1][i][1]) <= 1): 
                # if (sample[0][i][0] == sample[1][i][0]) and (sample[0][i][1] == sample[1][i][1]): # exact match code
                    partial_both.append(1.0) 
            score = sum(partial_both) / len(sample[0])
            total_both.append(score)

    return sum(total_both) / len(total_both)

def __main__():
    # replace decode_folder below with the appropriate folder 
    path = '/projectnb/ivc-ml/aburns4/MoTIF/task_automation/seq2act/decode/motif/su_all_9_5_step/decodes.joint_act'
    res = get_decode_results(path)
    gt_and_pred = get_gt_and_pred(res)
    action_gt_pred = get_action_gt_and_pred(gt_and_pred)
    ground_gt_pred = get_ground_gt_and_pred(gt_and_pred)
    both_gt_pred = get_both_gt_and_pred(gt_and_pred)

    action_comp = get_action_accuracy(action_gt_pred) * 100
    action_part = get_action_accuracy(action_gt_pred, complete=False) * 100
    print('Mean complete / partial action accuracy = %.1f / %.1f' % (action_comp, action_part))

    ground_comp = get_ground_accuracy(ground_gt_pred) * 100
    ground_part = get_ground_accuracy(ground_gt_pred, complete=False) * 100
    print('Mean complete / partial grounding accuracy = %.1f / %.1f' % (ground_comp, ground_part))

    both_comp = get_both_accuracy(both_gt_pred) * 100
    both_part = get_both_accuracy(both_gt_pred, complete=False) * 100
    print('Mean complete / partial both accuracy = %.1f / %.1f' % (both_comp, both_part))

__main__()
