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
    Input: X dataframe and impute_method to imput missing values (string)
    Output: updated dataframe with missing values imputed based on method desired
    """
    try:
        if (impute_method == 'Zero'):
            X.fillna(0,inplace=True)
        elif (impute_method == 'Mean'):
            X.fillna(X.mean(),inplace=True)
        else:
            X.fillna(X.median(),inplace=True) 
    except Exception as e:
        print (e)
    
        return X

# Checkpoint 5
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """
    Input: 
    Output: 
    """
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    return output_str, out_dict

# Checkpoint 6
def split_dataset(X, number):
    """
    Input: 
    Output: 
    """
    train=[]
    test=[]
    return train, test

# Restore Dataset
df = restore_dataset()

if df is not None:

    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe
    st.write(df)

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    st.markdown('Number of categories with missing values: {0:.2f}'.format(df.isna().any(axis=0).sum()))
    st.markdown('Average number of missing values per category: {0:.2f}'.format(df.isna().any(axis=0).sum()))
    st.markdown('Total number of missing values: {0:.2f}'.format(df.isna().any(axis=0).sum()))
        

    ############################################# MAIN BODY #############################################

    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect
    select_features = st.multiselect("Select two or more features for correlation", numeric_columns)

    # Remove the features using the remove_features function
    dropped_df= remove_features(df,select_features)

    # Display updated dataframe
    st.write('Updated dataframe with selected columns dropped')
    st.dataframe(dropped_df)

    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    select_impute_option = st.selectbox("Select way to impute missing values", ['Zero', 'Mean', 'Median'])

    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    imputed_df = impute_dataset(df,select_impute_option)
    
    # Display updated dataframe
    st.write('Updated dataframe with NaN values replaced with ', select_impute_option)
    st.dataframe(imputed_df)
    
    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')

    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect

    # Provide option to select multiple descriptive statistics to show using Streamit multiselect
 
    # Compute Descriptive Statistics including mean, median, min, max
    # ... = compute_descriptive_stats(...)
        
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')

    # Compute the percentage of test and training data
    
    # Print dataset split result

    # Save state of train and test split
