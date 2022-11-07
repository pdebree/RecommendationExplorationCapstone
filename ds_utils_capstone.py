# pandas and numpy are used for data format, in order to allow for easier manipulation.
import pandas as pd
import numpy as np

# seaborn and variables matplotlib packages are used for visualiations.
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Scipy's Stats gives us access to basic statistical analysis packages
from scipy import stats

# sklearn's OneHotEncoder allow us to encode categorical variables into dummy variables.
from sklearn.preprocessing import OneHotEncoder



def read_csv_pd(filepath, separator=",", index=False):
    """
    Converts a csv file, from a filepath, to a pandas DataFrame
    
    Creates, and returns, the DataFrame and prints the shape of the created DataFrame and 
    whether duplicated or missing values were found (usedin no_duplicates_missing). This functionality is 
    useful when reading in known, cleaned files to ensure they take the expected form.
    
    Parameters
    ----------
    filepath: string
                a filepath to the location of the .csv file to be read in. 
    
    separator: string 
                a string of the character used to separate data points
    
    index: boolean
                indicates whether the first column of the csv file should be read as an index
    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame of the csv file 
                
                
    
    Examples
    --------
    >>> read_csv_pd('data/clouds.csv')
    df
    
    See Also
    --------
    no_duplicates_missing: reports on whether missing values or duplicates were found in the DataFrame 
    """

    #assert 

    # check whether the index should be read in as the first column
    if (index==True):
        # read in csv file with index as the first column
        df = pd.read_csv(filepath, sep=separator, index_col=[0])
    else:
        # read in csv file with no index
        df = pd.read_csv(filepath, sep=separator, index_col=None)

    # print the shape of the created DataFrame
    print(f'DataFrame contains {df.shape[0]} rows and {df.shape[1]} columns.')

    # check if missing values or duplicates were found
    if (no_duplicates_missing(df) == False):
        # if found, print following message
        print('Missing values or duplicated rows found.')
    else:
        print("No missing values or duplicared rows found.")

    return df 

def no_duplicates_missing(df, report=False):
    """
    Takes in a recently read-in pandas dataframe and returns True if there are no duplicates and 
    no missing values. 
    
    Parameters
    ----------
    df : pandas DataFrame
                The DataFrame to be checked for missing or duplicated values. 
    report: boolean
                Whether or not the number of missing values should be reported.
    
    Returns
    -------
    found_dups: boolean
                A boolean value of whether no missing values were found and no duplicates were found
    
    Examples
    --------
    >>> no_duplicates_missing(df, report=True)
    5 missing values and 2 duplicate rows found.
    
    """

    # check whether duplicates and missing values 
    if (report==True):
        print(df.isna().sum().sum(), "missing values and",
              df.duplicated().sum().sum(), "duplicate rows found")

    # returns boolean of whether either missing or duplicated values were found
    found = (df.isna().sum().sum() == 0) & (df.isnull().sum().sum() == 0)  

    return found   


def cat_var(df):
    """
    Takes a dataframe of categorical variables and returns distribution information.
    
    Reports the the value_counts for each category, if there are less than 20 unique values, and
    a barplot for the values of each category.
    
    Paratmeters
    -----------
    df : Pandas DataFrame
            dataframe of categorical variables.
    
    Returns
    -------
    No return value but printed value counts and visualisations
    
    Examples
    --------
    >>> cat_var(df)
    ** plot of all variables in df **
    
    Notes
    -----
    All columns in dataset must be categorical.
    Utilises seaborn's countplot to displot counts of variables in all categories.
    
    See Also
    -------
    num_var: similar function but for numeric variables. 
    """


    # Creates list of the dataframe columns
    variables = df.columns

    # Loops over the names of the dataframe columns
    for i in variables:
        # if there are less than 20 values print value count and show count plot
        if df[i].nunique() < 20:
            print(df[i].value_counts().to_frame()) 
            # Initialise barchart
            plt.figure()
            # plot a count plot of the unique categories and how often they appear
            sns.countplot(x=df[i], order = df[i].value_counts().index)
            plt.ylabel("Count") # use count as y label 
            plt.xlabel(i) # use column name as x label 
            plt.xticks(rotation=40) # format ticks
            plt.show() # show plot 
        # if more than 20 variables, report the number.
        else:
            print(f'Number of Unique {i} Values: {df[i].nunique()}')



def num_var(df, bins=20):
    """
    Takes a dataframe of numeric variables and returns distribution information.
    
    Reports the the summary statistics and a histogram of the distribution.
    
    Paratmeters
    -----------
    df : Pandas DataFrame
            DataFrame of numeric variables.
    bins : integer
            The number of bins to be included in the histogram (number of groups).
    
    Returns
    -------
    No return value printed summary statistics and visualisations
    
    Examples
    --------
    >>> num_var(df)
    ** plot of all variables in df **
    
    Notes
    -----
    All columns in dataset must be numeric (date values can be included but this must be established
        within the DataFrame passed in).
    Utilises seaborn's countplot to displot counts of variables in all categories.
    
    See Also
    -------
    num_var: similar function but for numeric variables. 
    """


    # Creates a list of column names
    variables = df.columns

    # loops over column names
    for i in variables:
        print(i,'Summary Statistics:')

        # shows description of variable distributions
        print(df[i].describe(datetime_is_numeric=True)) 

        # Initialise histogram 
        plt.figure()
        sns.histplot(x=df[i], bins=bins)

        # Include mean and median in reporting.
        plt.axvline(np.median(df[i]), color='blue', label="Median")
        plt.axvline(np.mean(df[i]), color='red', label="Mean")

        # Include legend 
        plt.legend()
        # Rotation x labels for easier visability 
        plt.xticks(rotation=40)
        plt.show()



def equal_transform(df1, df2):
    """
    Takes in two series and returns whether they have equal value counts. 
    
    Useful when changing the value markers in columns, e.g. changing a Boolean series to a dummy series. 
        
    
    Paratmeters
    -----------
    df1 : Pandas Series (column of DataFrame)
            Series of values
    df2 : Pandas Series (column of DataFrame)
    
    
    Returns
    -------
    No return value but prints whether series are equal 
    
    Examples
    --------
    >>> equal_transform(df['color'], df['color_encoded'])
    Series are equal.
    
    
    See Also
    -------
    pandas.assert_series_equal : compares the value counts of series so differences in actual values are not an issue
    """


    # put the code here

    # Tries to check that the series are equal
    try: 
        pd.testing.assert_series_equal(df1.value_counts(),
                                 df2.value_counts(),
                                 check_names=False, check_index=False)
        print("Series are equal.")

    # If unable to assert that the series are equal, prints the following
    except:
        print("Series are not equal.")



def num_cat_cols(df, datetime_is_numeric=False):
    """
    Create a lists of numeric and of categorical variables for the passed in dataframe. 
    
    Has the functionality of being able to chose if datetime is numeric or not (this can depend 
    on situations and number of unique appearances of datetime). Default is datetime is not numeric.
   
        
    Paratmeters
    -----------
    df: Pandas DataFrame
            A DataFrame with columns.
            
    date_time_is_numeric : boolean
            A boolean to 
    
    Returns
    -------
    num_col : list of strings
                List of names of numerical columns
    
    cat_col : list of strings
                List of names of categorical columns
    
    Also, prints the names of these columns
    Examples
    --------
    >>> numeric, categorical = num_cat_cols(student_info)
    The Numeric columns: 
        age,
        score
    
    The Categorical columns: 
        gender,
        class
        
        
    See Also
    -------
    pandas.select_dtypes : selects the columns in a dataframe with the passed in datatype
    """


    # checks whether datetime variables should be considered as numeric
    if datetime_is_numeric==True:
        # makes a list of the names of the numeric columns (including datetime)
        num_col = list(df.select_dtypes(["number","datetime64"]).columns)
        # makes a list of the names of categorical columns 
        cat_col = list(df.select_dtypes("object").columns)
    else:
        # makes a list of the names of numeric columns (not including datetime)
        num_col = list(df.select_dtypes("number").columns)
        # makes a list of the names of categorical columns 
        cat_col = list(df.select_dtypes(["object","datetime64"]).columns)


    # prints the names of the categorical and numeric columns
    # .join iterates over the passed in list and prints it with the string it is passed on 
    print("The Numeric columns: \n\t", ",\n\t".join(num_col), sep="")
    print("")
    print("The Categorical columns: \n\t", ",\n\t".join(cat_col), sep="")

    # returns list of numeric columns and list of categorical columns 
    return num_col, cat_col




def ohe_sparse(sr):
    """
    Creates encodings for a numpy series and creates a sparse matrix of the encoded values.
    
    Uses sklearn's OneHotEncoder to turn a series of categorical values to a matrix of 
    the data rows and the categories as columns, with binary values (0 or 1) indicating 
    whether a row is associated with a particular category
        
    Paratmeters
    -----------
    sr: Pandas Series
            A Pandas series with categories.
    
    Returns
    -------
    df : Pandas DataFrame
            A Pandas DataFrame of the sparse encoding matrix category names as columns
    Examples
    --------
    >>> ohe_sparse(student_info['gender'])
    
    See Also
    -------
    sklearn.OneHotEncoder : 
    """

    # Initialise OneHotEncoder
    ohe = OneHotEncoder()

    # Fit the encoding and transform the data (needs to be transformed to a one dimensional numpy array
    encoded = ohe.fit_transform(np.array(sr).reshape(-1,1))                      

    # Creating a dense version of the matrix, to be added to a dataframe
    sparse_matrix = encoded.toarray()

    # Create dataframe version of dense matrix - uses original series index for index and values in series as columns
    df = pd.DataFrame(sparse_matrix, columns=ohe.categories_[0], dtype=int)

    # returns dataframe of encoded matri
    return df         