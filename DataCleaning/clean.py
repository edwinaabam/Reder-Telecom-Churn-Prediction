import pandas as pd
import ast
from sklearn.preprocessing import OneHotEncoder, LabelEncoder




def convert_dtype(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the data type of a specified column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column name to convert.
    dtype (str): The target data type ('int', 'float', 'str', 'datetime', 'list', 'dict').

    Returns:
    pd.DataFrame: The DataFrame with the converted column.
    """

    col_to_date = ['Timestamp','Start_Date', 'End_Date', 'FirstInteractionDate','LastInteractionDate',
                   'FirstActionTime','LastActionTime','most_recent_action_date','LastEmailSentDate',
                   'LastEmailOpenedDate','LastEmailClickedDate']
    
    for col in col_to_date:
        data[col] = pd.to_datetime(data[col], errors='coerce')

    return data
    

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
       engineer some features

        Returns:
        pd.DataFrame: The DataFrame with new features.
        """
    date_col_to_engineer = data.select_dtypes(include = 'datetime64').columns.tolist()  
    
    for col in date_col_to_engineer:
        date[col+'_year'] = date[col].dt.year
        date[col+'_month'] = date[col].dt.month
        date[col+'_day'] = date[col].dt.day

        data.drop(columns=[col], inplace=True)
        
    return data


def drop_irrelevant_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Drops irrelevant columns"""
    data.drop(columns = ['Name', 'Email', 'Phone',
                         'Address', 'Comment', 
                         'Timestamp'], inplace = True)
    
    return data
    
    
    
def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    ''' encodes data'''
    nominal_columns = ['Gender', 'TotalInteractionType', 'Frequency']
    ordinal_columns = ['Segment', 'Plan','customer_segments']
    cardinal_columns = ['Location',  'ProductList', 'MostCommonAction', 
                'LastPageVisited', 'FirstPageVisited', 'LastActionType', 'LeastFrequentAction']

    ohe = OneHotEncoder(sparse_output=False)

    encode_data = data.copy()

    encoded_columns = []

    for col in nominal_columns:
        transformed_col = ohe.fit_transform(encode_data[[col]])
        transformed_df = pd.DataFrame(transformed_col, columns = [f"{col}_{cat}" for cat in ohe.categories_[0]])
        
        encoded_columns.append(transformed_df)
        
    encode_data = pd.concat([encode_data] + encoded_columns, axis = 1)
    encode_data.drop(columns = nominal_columns, inplace = True)

    
    
    le = LabelEncoder()

    for col in ordinal_columns:
        encode_data[col] = le.fit_transform(encode_data[col])
        
    for col in cardinal_columns:
        encode_data[col] = encode_data.groupby(col)['TotalPurchaseValue'].transform('mean')
        
    encode_data.to_csv('encoded_data2.csv', index = False)
        
    return encode_data

        

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    '''Calls all the function above'''
    
    data = convert_dtype(data)
    data = feature_engineering(data)
    data = drop_irrelevant_columns(data)
    data = encode_data(data)
    
    return data
    




   