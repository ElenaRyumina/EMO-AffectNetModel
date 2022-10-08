def sequences(all_path, all_feature, win = 10, step = 5):  
    seq_path = []
    seq_feature_AN = []
    for id_cur in range(0, len(all_path)+1, step):
        need_id = id_cur+win
        curr_path = all_path[id_cur:need_id]
        curr_FE_AN = all_feature[id_cur:need_id].numpy().tolist()
        if len(curr_path) < win and len(curr_path) != 0:
            curr_path.extend([curr_path[-1]]*(win - len(curr_path)))
            curr_FE_AN.extend([curr_FE_AN[-1]]*(win - len(curr_FE_AN)))
        if len(curr_path) != 0:
            seq_path.append(curr_path)
            seq_feature_AN.append(curr_FE_AN)
    return seq_path, seq_feature_AN


def df_group(df, label_model):
    df_group = df.groupby(['frame']).agg({i:'mean' for i in label_model})

    df_group.reset_index(inplace=True)
    df_group = df_group.sort_values(by=['frame'])
    df_group.reset_index(drop=True)
    return df_group