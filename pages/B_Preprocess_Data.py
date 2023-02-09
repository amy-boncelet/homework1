import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

# Checkpoint 1
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    if 'house_df' not in st.session_state:
        data = st.file_uploader("Upload your data", type=['csv','txt'])
        if data:
            df = pd.read_csv(data)
            st.session_state['house_df'] = df
        else:
            return None
    else:
        df = st.session_state['house_df']
    return df

# Checkpoint 3
def remove_features(X,removed_features):
    """
    Input: full dataframe X and list of columns, removed_features, to remove
    Output: new dataframe with dropped columns
    """
    dropped_X = X.drop(columns=removed_features)
    
    return dropped_X

# Checkpoint 4
def impute_dataset(X, impute_method):
    """
    Input: X (dataframe) with missing values and impute_method (string) use to imput missing values
    Output: updated dataframe with missing values imputed based on method desired
    """
    # try:
    if (impute_method == 'Zero'):
        imputed_df = X.fillna(0)
    elif (impute_method == 'Mean'):
        imputed_df = X.fillna(X.mean())
    else:
        imputed_df = X.fillna(X.median()) 
    # except Exception as e:
    #     print (e)
    
    return imputed_df

# Checkpoint 5
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """
    Input: 
    Output: 
    """
    for feature in stats_feature_select: 
        output_str=feature
        out_dict = {
            'mean': X[feature].mean(),
            'median': X[feature].median(),
            'max': X[feature].max(),
            'min': X[feature].min()
        }
        for stat in stats_select:

            output_str = output_str + stat

    return output_str, out_dict

# Checkpoint 6
def split_dataset(X, number):
    """
    Input: 
    Output: 
    """
    try: 
        number = int(number)/100
        train, test = train_test_split(X, test_size=number)
    except Exception as e:
        print(e)

    return train, test

# Restore Dataset
df = restore_dataset()

if df is not None:

    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe
    st.write(df)

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    st.markdown('Number of categories with missing values: {0:.2f}'.format(df.isna().any(axis=0).sum()))
    st.markdown('Average number of missing values per category: {0:.2f}'.format(df.isna().sum().mean()))
    st.markdown('Total number of missing values: {0:.2f}'.format(df.isna().sum().sum()))
        

    ############################################# MAIN BODY #############################################

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect
    select_features = st.multiselect("Select features to remove from dataset", numeric_columns)

    # Remove the features using the remove_features function
    dropped_df= remove_features(df,select_features)

    # Display updated dataframe
    st.markdown('#### Updated dataframe with selected columns dropped')
    st.dataframe(dropped_df)

    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    select_impute_option = st.selectbox("Select way to impute missing values", ['Zero', 'Mean', 'Median'])
    
    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    imputed_df = impute_dataset(df,select_impute_option)
    
    # Display updated dataframe
    st.dataframe(imputed_df)
    
    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')

    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect
    descriptive_stat_features = st.multiselect("Select columns to show descriptive statistics", numeric_columns)

    # Provide option to select multiple descriptive statistics to show using Streamit multiselect
    descriptive_stat_options = st.multiselect("Select descriptive statistics to show", ['Mean', 'Median', 'Min', 'Max'])
 
    # Compute Descriptive Statistics including mean, median, min, max
    # (feature_str, stat_dict) = compute_descriptive_stats(df,  descriptive_stat_features,  descriptive_stat_options)
    # for feature in feature_str:
    #     st.write(feature, )
        
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')
    test_perc = st.text_input('Enter size of test set (%)', '25')

    # Compute the percentage of test and training data
    (train, test) = split_dataset(df,test_perc)
    train_size = train.shape[0]
    test_size = test.shape[0]
    total_size = test_size + train_size
    train_perc = (train_size/total_size)*100
    
    # Print dataset split result
    st.write(f'The complete dataset contains {total_size} observations. The training dataset contains {train_size} observations ({train_perc}% of data). The testing dataset contains {test_size} observations ({test_perc}% of data)')
    #st.write('Train size is :', train_size)

    # Save state of train and test split
    st.session_state['train_df'] = train
    st.session_state['test_df'] = test
