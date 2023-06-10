import pandas as pd
import numpy as np

# Performs leakage between one numeric column and numeric target
def leakages_numeric_numeric(data, c, y, p):
    for i in range(len(data)):
        r = np.random.uniform(0, 1)
        if r >= (1 - p):
            # Modifies the value in column 'c' based on the leakage formula
            data.loc[i, c] = (1 - p) * data[c].iloc[i] + p * data[c].iloc[i]
    return data

# Performs leakage between a categorical column and numeric target
def leakages_categorical_numeric(data, c, y, p):
    # Drops rows with missing values
    data = data.dropna()
    # Determines the number of bins based on the unique values in column 'c'
    num_bins = len(data[c].unique())
    # Creates bins for column 'y' based on the unique values in column 'c'
    bins = pd.cut(data[y], num_bins, labels=data[c].unique(), duplicates='drop')
    for i in range(len(data)):
        r = np.random.uniform(0,1)
        if r >= (1-p):
            # Updates the value in column 'c' with the corresponding bin value
            data.loc[i,c] = bins.iloc[i]
    return data

# Creates a list of unique categories in column 'c'
def defining_cat_list(data, c, y):
    categories = data[c].unique()
    # Creates a list of unique targets in column 'y'
    targets = data[y].unique()
    cat_list = []
    # Iterates over each category
    for cat in categories:
        aux_dict = {}
        aux_dict['category'] = cat
        # Filters the data for the current category
        aux_df = data[data[c] == cat]
        # Calculates the count of occurrences for each target in the filtered data
        counts = aux_df[y].value_counts()
        aux_dict['count'] = sum(counts)
        # Stores the count of each target in the dictionary
        for t in targets:
            aux_dict[t] = counts[t] if t in counts.keys() else 0
        cat_list.append(aux_dict)
    # Converts the list of dictionaries into a DataFrame
    cat_list = pd.DataFrame(cat_list)
    # Calculates the encoded target value for each category
    for t in targets:
        cat_list[c + '_encoded_target_' + str(t)] = cat_list[t] / cat_list['count']
    return cat_list

# Performs leakage between a categorical column and categorical target
def leakages_categorical_categorical(data, c, y, p):
    # Creates a DataFrame with category counts and encoded target values
    cat_list = defining_cat_list(data, c, y)
    # Creates a copy of the original data
    df_aux_f = data.copy()
    # Iterates over each row in the data
    for i in range(len(data[c])):
        p_uv = np.random.uniform(0, 1)
        if p_uv >= (1 - p):
            # Retrieves the current category and target values
            cat = data.loc[i, c]
            tg = data.loc[i, y]
            # Constructs the column name for the encoded target value
            st = c + "_encoded_target_" + str(tg)
            # Finds the corresponding row in the cat_list DataFrame
            aux = cat_list.loc[cat_list['category'] == cat]
            # Updates the value in the copied data with the encoded target value
            df_aux_f.at[i, c] = float(aux[st].values[0])
    return df_aux_f

#Performs leakage between a numeric column and categorical target
def leakages_numeric_categorical(data, c, y, p):
    # Groups the data by the target column and calculates the mean of column 'c'
    group_c = data.groupby(y)[c].mean()
    group_c = group_c.to_frame()
    group_c = group_c.reset_index(drop=False)
    # Performs leakage between a numeric column and a categorical column
    for i in range(len(data)):
        p_uv = np.random.uniform(0,1)
        if p_uv >= (1-p):
            # Retrieves the mean value for the corresponding target
            aux = group_c.loc[group_c[y] == data[y].iloc[i]][c]
            # Updates the value in column 'c' with the mean value
            data.at[i,c] = aux.values[0]
    return data

def introducing_leakages_function(data_leakage, c, target, p, attrs_categoric, classf):
    if classf == 1:
        if c in attrs_categoric:
            print("entrei aqui")
            # Creates a dictionary to map unique categorical values to indices
            unique_dict = {}
            for i in range(len(data_leakage[c].unique())):
                unique_dict[data_leakage[c].unique()[i]] = i
            # Performs leakage between categorical columns
            data_leakage = leakages_categorical_categorical(data_leakage, c, target, p)
            # Maps the updated categorical values back to their original indices
            for i in range(len(data_leakage[c])):
                if data_leakage[c][i] in unique_dict.keys():
                    data_leakage.at[i, c] = unique_dict[data_leakage[c].iloc[i]]
                else:
                    data_leakage.at[i, c] = data_leakage[c].iloc[i]
            # Converts the column type to float
            data_leakage[c] = data_leakage[c].astype(float)
        else:
            # Performs leakage between a numeric column and a categorical column
            data_leakage = leakages_numeric_categorical(data_leakage, c, target, p)
    elif classf == 0:
        if c in attrs_categoric:
            # Performs leakage between a categorical column and a numeric column
            data_leakage = leakages_categorical_numeric(data_leakage, c, y, p)
        else:
            # Performs leakage between two numeric columns
            data_leakage = leakages_numeric_numeric(data_leakage, c, y, p)
    return data_leakage
