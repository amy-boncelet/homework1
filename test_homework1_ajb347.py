from pages import A_Explore_Dataset, B_Preprocess_Data
import pandas as pd
import numpy as np
import streamlit as st
import plotly as pl
import matplotlib.pyplot as plt

# ask user to inport a csv dataset
st.header('Import Dataset')
st.subheader('Upload a Dataset')
data = st.file_uploader("Upload a csv file")
df = pd.read_csv(data)
st.dataframe(df.head())
num_features=list(df.select_dtypes(['float', 'int']).columns) # extract numerical columns from dataframe
st.write(num_features)

st.header('Visualize Features')

def user_input_features():
        st.sidebar.text ("Select type of chart")
        chart = st.sidebar.selectbox("Type of chart",("Scatterplot", "Lineplot", "Histogram", "Boxplot"))

        st.sidebar.text ("Specify Input Parameters")
        x_axis = st.sidebar.selectbox("Select x-axis", df.columns)
        y_axis = st.sidebar.selectbox("Select y-axis", df.columns)

        st.sidebar.text("Filter features")
        for column in num_features: # iterate through each column in the dataframe that is numerical
            feature = st.sidebar.slider(column, float(df[column].min()), float(df[column].max())) # add slider feature to sidebar

        return (chart, x_axis, y_axis)

(chart, x_axis, y_axis) = user_input_features()

fig, ax = plt.subplots()
ax.set_title(chart + ' plotting the '+ x_axis + ' by the ' + y_axis)
ax.set_xlabel(x_axis)
ax.set_ylabel(y_axis)

if chart == 'Scatterplot':
    ax.scatter(df[x_axis], df[y_axis])
elif chart == 'Lineplot':
    ax.plot(df[x_axis], df[y_axis])
elif chart == 'Histogram': 
    ax.hist(df, bins=20)
else:
    ax.boxplot(df)
        
st.pyplot(fig)

######## CheckPoint1 ##############
student_filepath="datasets/housing/housing.csv"
test_filepath= "test_dataframe_file/inital_housing.csv"
s_dataframe = pd.read_csv('C:\\Users\\abonc\\OneDrive\\Documents\\GitHub\\homework1\\datasets\\housing\\housing_paml.csv')
e_dataframe = pd.read_csv('C:\\Users\\abonc\\OneDrive\\Documents\\GitHub\\homework1\\datasets\\housing\\housing.csv')
e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
numeric_columns = list(e_X.select_dtypes(['float', 'int']).columns)
nan_colns = e_X.columns[e_X.isna().any()].tolist()

st.write('text in checkpoint 1')

st.dataframe(s_dataframe.head())

st.header("Where are the most expensive properties located?")
st.subheader("On a map")
st.markdown("The following map shows the most expensive areas in Californai, with median home values $400,000 and above")
st.map(s_dataframe.query("median_house_value>=400000")[["latitude", "longitude"]].dropna(how="any"))
st.subheader("In a table")
st.markdown("Following are the top five most expensive properties.")
st.write(s_dataframe.query("median_house_value>=400000").sort_values("median_house_value", ascending=False).head())

st.subheader("Selecting a subset of columns")
#st.write(f"Out of the {s_dataframe.shape[1]} columns, you might want to view only a subset. Streamlit has a [multiselect](https://streamlit.io/docs/api.html#streamlit.multiselect) widget for this.")
defaultcols = ["housing_median_age", "total_rooms", "total_bedrooms", "population", "households"]
cols = st.multiselect("Columns", s_dataframe.columns.tolist(), default=defaultcols)
st.dataframe(s_dataframe[cols].head(10))

def test_load_dataframe():
    s_dataframe = A_Explore_Dataset.load_dataset(student_filepath)
    e_dataframe =pd.read_csv(test_filepath)
    pd.testing.assert_frame_equal(s_dataframe,e_dataframe)

###################################

st.header('Feature Correlation')

######## CheckPoint2 ##############
## You have to round to two decimal places
def test_compute_descriptive_stats():
    _, out_dict=B_Preprocess_Data.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.26,
        'max': 41.95,
        'min': 32.54
    }
    assert out_dict==e_dict
    

###################################


######## CheckPoint3 ##############
def test_compute_corr():
    e_corr = np.array([[1, -0.0360996], [-0.0360996, 1]])
    test_corr = A_Explore_Dataset.compute_correlation(e_dataframe,['latitude','total_rooms'])
    print(test_corr)
    print(e_corr)
    assert np.allclose(e_corr, test_corr)
    # assert test_corr == 

###################################


######## CheckPoint4 ##############

def test_impute_zero():
    e_zero_df = pd.read_csv("test_dataframe_file/Zero.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_zero_df = B_Preprocess_Data.impute_dataset(e_X, 'Zero', numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_zero_df,s_zero_df)


def test_impute_median():
    e_median_df = pd.read_csv("test_dataframe_file/Median.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_median_df = B_Preprocess_Data.impute_dataset(e_X, 'Median', numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_median_df,s_median_df)


def test_impute_mean():
    e_mean_df = pd.read_csv("test_dataframe_file/Mean.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_mean_df = B_Preprocess_Data.impute_dataset(e_X, 'Mean', numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_mean_df,s_mean_df)


###################################


######## CheckPoint5 ##############

def test_remove_features():
    e_remove= pd.read_csv("./test_dataframe_file/remove.csv")
    e_X= e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_remove = B_Preprocess_Data.remove_features(e_X, ['latitude', 'longitude'])
    pd.testing.assert_frame_equal(s_remove,e_remove)

###################################


######## CheckPoint6 ##############

def test_split_dataset():
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_split_train, s_split_test = B_Preprocess_Data.split_dataset(e_X,30)

    assert s_split_train.shape == (14448, 9)

###################################
