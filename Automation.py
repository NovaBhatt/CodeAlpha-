import streamlit as st
import pandas as pd

st.set_page_config(layout='wide', page_title='Data Cleaning')

st.title("Data Cleaning")
# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    print(df)

    # Drop duplicate rows from the DataFrame
    df.drop_duplicates(inplace=True)

    # Identify columns with missing values and store them in a list
    columns_with_na = df.columns[df.isna().any()].tolist()

    # Print the columns that have missing values
    st.write("The columns in your provided database with missing values are: ", columns_with_na)

    # Initialize an empty list to store actions and a variable for indexing
    action = []
    i = 0

    # Loop through each column with missing values to get user input on how to handle them
    while i < len(columns_with_na):
        action_input = st.selectbox(f"What do you want to do with the missing values in {columns_with_na[i]}?",
                                    options=['delete', 'fill with custom', 'fill with median', 'fill with mean', 'fill with mode'],
                                    key=f"action_{i}")
        action.append(action_input)
        i += 1

    # Apply the specified actions to the DataFrame
    for i, col in enumerate(columns_with_na):
        if action[i] == "delete":
            df.drop(col, axis=1, inplace=True)  # Drop the column if 'delete' is chosen
        elif action[i] == "fill with custom":
            custom = st.text_input(f"Please enter your custom value for {col}: ", key=f"custom_{i}")
            df[col].fillna(value=custom, inplace=True)  # Fill missing values with a custom value
        elif action[i] == "fill with median":
            df[col].fillna(value=df[col].median(), inplace=True)  # Fill missing values with the median
        elif action[i] == "fill with mean":
            df[col].fillna(value=df[col].mean(), inplace=True)  # Fill missing values with the mean
        elif action[i] == "fill with mode":
            df[col].fillna(value=df[col].mode()[0], inplace=True)  # Fill missing values with the mode

    # Print the updated DataFrame
    st.write("Your new dataset is: ", df)

    # Select numeric columns from the DataFrame
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Loop through each numeric column to detect and handle outliers
    for col in numeric_cols:
        if df[col].notna().all():  # Ensure the column has no missing values
            Q1 = df[col].quantile(0.25)  # Calculate the first quartile
            Q3 = df[col].quantile(0.75)  # Calculate the third quartile
            IQR = Q3 - Q1  # Calculate the interquartile range
            lower_bound = Q1 - 1.5 * IQR  # Calculate the lower bound for outliers
            upper_bound = Q3 + 1.5 * IQR  # Calculate the upper bound for outliers
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]  # Identify outliers
            edit = st.radio(f"Outliers detected in column {col}. Should we delete them?", ('Yes', 'No'), key=f"edit_{col}")
            if edit == "Yes":
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]  # Remove outliers if user chooses 'yes'

    # Print the final DataFrame
    st.write("Your final dataset is: ", df)
