import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

plt.style.use('seaborn')

from scipy import stats
# The Stats model api gives us access to regression analysis packages 
import statsmodels.api as sm

#import ds_utils_capstone

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

import ds_utils_capstone

import os
from pydub import AudioSegment
import librosa 
import librosa.display



def artist_terms_formatting(df): 
    """

    
    
    Success! We can now look more deeply into `artist_term_frequency`. We can see that this a string that has newline characters in it form formatting in. 

There is also an issue of blank space after 1. values. We will remove the start and end characters for the list, remove the `n` characters and then split (the default being any whitespace, so this will deal with our odd lengths of whitespace).

Finally, we will apply this same lambda function to the `artist_terms_weight` string. 

Success! We have turned these three into lists, however, we still need to check that we could map them onto each other. For this they need to be the same length. (Consider matrix multiplication).
To do this we will get the lengths of the values of the three lists and ensure that they are the same values.
    
    
    
    """
    
    """
    Turns the artist_terms columns ('artist_terms', 'artist_frequency', 'artist_weights') in numpy arrays 
    
    Originally these variables were read in as strings, but for functionality they should be numpy arrays.
    Uses string splitting functions to create strings of each value (for terms) and floats (for frequencies
    and weights) and then makes a list of this. 
    
    Removes the start and end characters from the lists, then any formatting characters (\ and \n) and 
    splits the terms by ', ' (the value between actual terms) and the freqs and weights by whitespace (this
    deals with inconsistent whitespaces as well).
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the artist terms columns as strings
    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with the artist terms columns as numpy arrays
                
                
    Examples
    --------
    >>> artist_terms_formatting(df)
    df
    """
    
    # takes the series of strings, strips the brackets, removes \ values (from formatting) and splits by ', '
    artist_terms_list = df['artist_terms'].apply(
        lambda x: np.array(x.strip('"[\'\']"').replace('\'','').split(', ')))
    
    # takes the series of strings, strips the brackets, removes \n values (from formatting) and splits on empty space
    artist_terms_freq_list = df['artist_terms_freq'].apply(
        lambda x: np.array(list(map(float, x.strip('[\'\']').replace('\n','').split()))))
    
    # apply the same logic to the weights
    artist_terms_weight_list = df['artist_terms_weight'].apply(
        lambda x: np.array(list(map(float, x.strip('[\'\']').replace('\n','').split()))))
                                                  
    # create dataframe with the three numpy arrays series as columns
    terms_lengths = pd.DataFrame({'terms':artist_terms_list , 'freqs':artist_terms_freq_list, 
                              'weights':artist_terms_weight_list})
    
    
    
    # prints the head of the dataframe where
    print(terms_lengths[~(terms_lengths['terms'].map(len) == terms_lengths['freqs'].map(len))].head(2), '\n')
    
    # prints the number of rows where the empty
    print("Number of Empty Artist Terms Artists:", 
          len(terms_lengths[~(terms_lengths['terms'].map(len) == terms_lengths['freqs'].map(len))]))
    
    # replace original columns with updated columns, from terms_length dataframe
    df['artist_terms'] = terms_lengths['terms']
    df['artist_terms_freq'] = terms_lengths['freqs']
    df['artist_terms_weight'] = terms_lengths['weights']
    
    return df


def binning_variables(sl):
    """
    Performs binning on selected variables from the lakh midi dataset. 
    
    Creates decided bins to create categorical variables from numeric. Original variable are dropped
    and replaced with binned versions. More details of decision making can be found in `2_EDA`. 
                
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the artist terms columns as strings
    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with the artist terms columns as numpy arrays
                
                
    Examples
    --------
    >>> artist_terms_formatting(df)
    df
    
    Notes
    -----
    Specifics of binning as follows:
    
    'duration' binned to 'duration_bins' with:
        'short', 'regular', 'long'
        
    'tempo' binned to 'tempo_bins' with:
        '0-49bpm', '50-99bpm', '100-149bpm', '150-199bpm', '200+ bpm'
        
    'time_signature' turned to a binary variable:
        '4/4_time'
        
    'year' binned to 'year_bins' with: 
        'dec_40s/50s', 'dec_60s', 'dec_70s', 'dec_80s', 'dec_90s', 'dec_00s'
    
    'start_of_fade_out_prop' binned to 'fade_out_bins' with:
        'long_fade','fade','no_fade'
        
    'key' turned to a binary variable:
        'popular_key'
    
        
    
    See Also
    --------
    feat_encoding : performs overall feature encoding for dataframe (calls this function)
    """

    
    # create bins for duration by using the cut function
    sl['duration_bins'] = pd.cut(sl['duration'],
                           bins=[0.0, 180.0, 300.0, np.inf],
                           labels = ['short', 'regular', 'long'])
    
    
    # create bins for tempo byusing the cut function
    sl['tempo_bins'] = pd.cut(sl['tempo'], 
                         bins=[0,49,99,149,199,np.inf],
                         labels= ['0-49bpm', '50-99bpm', '100-149bpm',
                                 '150-199bpm', '200+ bpm'],
                         include_lowest=True)

    
    # converting 4/4 time signatures to dummy variables (the inherent is that the time
    # signature is neither of these).
    sl['4/4_time'] = sl['time_signature'].map({4:1, 3:0, 1:0, 5:0, 7:0, 0:0})
    
    # create bins for song decade using the cut function
    sl['year_bins'] = pd.cut(sl['year'], 
                            bins=[1947, 1960, 1970, 1980, 1990, 2000, 2011],
                            labels = ['dec_40s/50s', 'dec_60s', 'dec_70s', 'dec_80s', 'dec_90s', 'dec_00s'])
    
    # create bins for fade out using the cut functions
    sl['fade_out_bins'] = pd.cut(sl['start_of_fade_out_prop'], 
                                 bins=[0,
                                       sl['start_of_fade_out_prop'].quantile(q=0.25),
                                       0.99,
                                       np.inf], 
                                 labels=['long_fade','fade','no_fade'])
    
    # creates binary variable by mapping popular keys to 1 and not popular keys to 0 
    sl['popular_key'] = sl['key'].map({'A':1, 'G':1, 'D':1, 'B':0, 'C#':0, 'C':1, 'F':0, 'F#':0, 
                                       'E':0, 'Ab':0, 'Bb':0, 'Eb':0})
    
    # removing the now binned variables
    sl = sl.drop(columns=['duration','tempo','time_signature','year',
                         'start_of_fade_out_prop', 'key'])
    
    
    # returns the updated dataframe
    return sl 
    


def add_list_matrix(df, list_column, mean_threshold=0.05):
    """
    Creates a matrix of artist term encoding for each row of the data and adds to passed in DataFrame
    
    Takes in a pandas DataFrame and column to be binarised. Uses sklearn's MultiLabelBinarizer function 
    to create a appearance matrix of the elements in the list_column's lists for each row. This is then added 
    back to the original dataframe. However, it removes values that appear less than a decided amount of 
    times in the matrix (passed in as mean_threshold). This reduces the size of the returned data and ensure that no
    irrelevant variables are included. Default appearance is 5% of the data.
    

    
    Parameters
    ----------
    df : Pandas DataFrame
                DataFrame with the list_column in it.
                
    list_column : string
                Name of the column to be binarised (column must be made up of numpy arrays)
    
    mean_threshold : float
                Minimum percentage of times that a variable must appear to be included

    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with the binarised variables merged with the original 

    Examples
    --------
    >>> add_list_matrix(df, 'shape' mean_threshold=0.1)
    df
    
    Note
    ----
    Original column is not removed in this stage
    
    See Also
    --------
    sklearn.MultiLabelBinarizer : 
    feat_encoding : performs overall feature encoding for dataframe (calls this function)
    """
        
    # Initialise multilabel binariser to take our column of lists to a dense matrix 
    mlb = MultiLabelBinarizer()
    
    # create dense matrix by fitting and transforming the selected column
    list_matrix = mlb.fit_transform(df[list_column])
    
    # create dataframe of matrix data using classes as column labels and original DataFrame index
    matrix_pd = pd.DataFrame(data=list_matrix, columns=mlb.classes_, 
                                  index=df.index)
    
    # calculate means for every column to get percentage appearance
    variable_means = matrix_pd.mean().sort_values(ascending=False)
    
    # select artist terms that appear more than the threshold
    relevant_variables = variable_means.loc[variable_means > mean_threshold].index.values

    df2 = pd.merge(df, matrix_pd[relevant_variables], 
                   left_index=True, right_index=True)
    
    return df2

    
    
def feat_encoding(sl, min_similar_artist_appearance=0, min_term_appearance=0):
    
    """
    Wrapper function for all encoding to be performed on the Lakh Midi Dataset
    
    Three distinct steps:
        1) Binning - Selected variables are binned by calling the binning_variables function
        2) Encoding - Binned Variables with more than 2 categories are encoded using
                        a OneHotEncoder within the ohe_sparse function
        3) Binarising Lists - Selected variables (containing lists) are turned to a appearance
                        matrices and added back to the dataframe.
                        
    Performs above steps and returns the transformed dataframe, as well as a list of the columns
    except those created by the binarisation. This list will allow users to decide which variable
    from the encodings to drop (by looking at correlations) and look at how correlated variables are
    overall.
    
    
    Parameters
    ----------
    sl: Pandas DataFrame
                DataFrame containing non-binned variables.
    
    min_similar_artist_appearance : float
                The minimum percentage for binarising the 'similar_artists' variable
        
    min_term_appearance : float
                The minimum percentage for binarising the 'similar_artists' variable

    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with all of the necessary values encoded 
    non_matrix_columns : list
                a list of columns that are not the result of binarisation 

    
    Examples
    --------
    >>> feat_encoding(df, min_similar_artist_appearance=0.05, min_term_appearance=0.05)
    df
    
    Note
    ----
    No dummy variables are dropping in the encoding to allow the user to chose. One variable must 
    be dropped from all binnings in order to avoid multicollinearity (the dropped variable becomes
    the inherent value of the particular categorical variable). 
    
    See Also
    --------
    no_duplicates_missing: reports on whether missing values or duplicates were found in the DataFrame 
    binning_variables : function used to created bins for selected variables
    ohe_sparse: function used to create encodings for non-binary binned variables
    add_list_matrix : function used to binarise lists columns
    
    """
    
    # create binnings for variables to be encoded.
    sl = binning_variables(sl)

    # create list of non-binary binned variables that need encoding
    cols_to_encode =  ['duration_bins', 'tempo_bins', 'year_bins', 'fade_out_bins']

    # loop over list of binned variable
    for i in cols_to_encode:
        # encode binned variable and add columsns to original dataframe 
        sl = pd.merge(sl, pd.DataFrame(data=ds_utils_capstone.ohe_sparse(sl[i])).set_index(sl.index.values),
                        left_index=True, right_index=True).drop(columns=[i])
    
    # drop unnecessary columns
    sl = sl.drop(columns = ['artist_terms_freq', 'artist_terms_weight', 'key_confidence', 
                            'mode_confidence', 'time_signature_confidence', 'sections_confidence', 'sections_start'])
 
    # create a list of columns without the binarisation
    non_matrix_columns = sl.drop(columns = ['artist_terms', 'similar_artists']).columns.values
    
    # add similar artist encodings for all artists who appear (in these lists) more than the chosen number of times
    sl = add_list_matrix(sl, 'similar_artists', min_similar_artist_appearance)
    
    # add similar artist encodings for all artists who appear (in these lists) more than the chosen number of times
    sl = add_list_matrix(sl, 'artist_terms', min_term_appearance)
    
    # remove unnecessary variables 
    sl = sl.drop(columns = ['artist_terms', 'similar_artists'])
    
    return sl, non_matrix_columns


def taste_profile_reduction(taste_profile, songs_in_dataset, min_user_appearance=10, 
                           min_song_appearance=5):
    """
    Creates a subset of the passed in taste profile depending on the song appearances
    in aligned dataset and minimum appearance distinctions.
    
    Removing some songs through EDA requires that the taste profile be updated to create 
    a representation of the user taste based (we will not want to carry unnecessary data 
    into our modelling). Also limits the users and songs to a minimum appearance value 
    to ensure that there is not too much unuseful data.

    We also want to ensure that a user listened to songs more than once - we cannot gain value 
    from their listening patterns if they only listened to all the songs they listened to one 
    time. 
   
    
    Parameters
    ----------
    taste_profile : Pandas DataFrame 
                A dataframe of the taste profile triplets (user, song, count)
                
    songs_in_dataset : Pandas Series
                A series of the songs in a dataset (song index of the lmd dataframe)
    
    min_user_appearance : float 
                A minimum number of times that a user must appear in the taste profile
                to be included. 
    min_song_appearance : float
                A minimum number of times that a song must appear in the taste profile
                to be included.

    
    
    Returns
    -------
    taste_profile_minapp: Pandas DataFrame
                a Pandas DataFrame of the correctly reduced taste profile
                
    Also, prints the new size and number of unique listeners and unique songs 


    Examples
    --------
    >>> taste_profile_reduction(tp, lmd_songs, min_user_appearance=10, 
                           min_song_appearance=5)
    Created Taste Profile with 1985026 overall row triplets,
         80965 Unique Users,
        6823 Unique Songs,
    and a minimum song appearance of 5 and user appearance of 10
       
       
    Notes
    -----
    As the user and song reduction are done at the same time, they may affect each other.
    If a user is in the list of enough listens, they will be included but then the songs 
    they listened to may not appear enough overall so may be removed. This would reduce
    their appearance count.As these both need to be done it is a good idea to overshoot 
    the necessary minimum appearance.
    
    See Also
    --------
    sklearn.MultiLabelBinarizer : 
    feat_encoding : performs overall feature encoding for dataframe (calls this function)
    """
        
    # select songs that are in the songs dataset (needs to be done first)
    taste_profile_rightsongs = taste_profile[taste_profile['song'].isin(songs_in_dataset)]
    
    # creates a series of counts for the number of times a user appears in the dataset. 
    uc = taste_profile_rightsongs['user'].value_counts()
    # creates array of users who appear more than the floor limit 
    enough_listen_users = uc[uc >= min_user_appearance].index.values
  
    # creates a series of counts for the number of times a song appears in the dataset. 
    sc = taste_profile_rightsongs['song'].value_counts()
    # creates array of songs that appear more than the floor limit 
    enough_listen_songs = sc[sc >= min_song_appearance].index.values
    
    # limiting the users to only those that appear enough times (based on min_appearance)
    taste_profile_minapp = taste_profile_rightsongs[(taste_profile_rightsongs['user'].isin(enough_listen_users))
                                 & (taste_profile_rightsongs['song'].isin(enough_listen_songs))]
    
    print('Created Taste Profile with', len(taste_profile_minapp),'overall row triplets,\n\t',
          taste_profile_minapp['user'].nunique(),'Unique', 'Users,\n\t', 
          taste_profile_minapp['song'].nunique(),'Unique Songs,\n\t',
          'and a minimum song appearance of', min_song_appearance, 'and user appearance of',
         min_user_appearance)
    
    
    return taste_profile_minapp


def taste_aggregation_features(df_song, df_tp):
    """
    Creates aggregations of total songs listens and average lists and adds them to the 
    song (LMD) dataframe. 
    
    Performs aggregations on the total number of time a song was listened, the total number
    of listeners who listened to that song and the average listen count (the average number
    of times a user listened to a particular song). Aggregate the taste profile song 
    and user appearances by song, so the song's id is used merge back to the original song data. 


    Parameters
    ----------
    df_song: Pandas DataFrame
                a Pandas DataFrame of the song data

    df_tp: Pandas DataFrame
                a Pandas DataFrame of the taste profile data
    
    Returns
    -------
    df_song_agg : Pandas DataFrame
                a Pandas DataFrame of the song data with the addition of the aggregation variables.
                
    Notes
    -----
    This function creates variables that are could be seen as overfitting to the data. However, if 
    doing only content based recommendation, these additions could be seen as extra exogenous data because 
    they could represent a general popularity measure of the songs (this builds on the assumption that the 
    user we are recommending to does not appear in the taste profile, as they are a new user). 
    
    Examples
    --------
    >>> taste_aggregation_features(sl, tp)
    df
    """
    
    # create dataframe of how many times each song was listened to 
    song_counts = df_tp.groupby(['song'])['count'].sum().to_frame()
    # create dataframe of how many users listened to each song
    listeners_counts = df_tp.groupby('song')['user'].count().to_frame().rename(columns={'user':'listeners_count'})
    
    # Add thse back to our main dataset by merging on index  
    df_song = df_song.merge(song_counts, left_index=True, right_on='song', how='inner').rename(
        columns={'count': 'total_listen_count'})
    df_song = df_song.merge(listeners_counts, left_index=True, right_on='song', how='inner')
    
    # Creating an average listens per song column by dividing total listens by number of users who have listened to the song.
    df_song['avg_listen_count'] = df_song['total_listen_count']/df_song['listeners_count']

    # returns the altered song dataset
    return df_song    

def pca_explore(df):
    
    """
    Performs Principal Component Analysis on the passed in dataframe
    
    Fit the full dataframe to sklearn's PCA class and evaluates the success of the dimensionality 
    reduction with the different component values. Hard coded to scale using a Min-Max Scaler because the 
    lmd encoded data has a lot of binary variables. Scaling is need to perform unbias dimension-based 
    analysis. Finally, shows plots of amount variance explained with diffent number of components.
    
    Parameters
    ----------
    df : Pandas DataFrame
                a Pandas DataFrame which will be analysed using PCA
                
    Returns
    -------
    plt : MatPlotLib Figure Object 
                a object carrying the specifics of the plot created, so it can be altered or shown in 
                the notebook 

    Examples
    --------
    >>> pca_explore(sl)
    plt
    
    Notes
    -----
    
    plot = pca_explore(sl)   
    plot.show()
    
    This shows the created plots in the notebook.
    User could also add additional titles or makes changes from here. 
    
    
    See Also
    --------
    sklearn.PCA: function used to perform PCA 
    """
    # scales the data using a minmax 
    df_scale = MinMaxScaler().fit_transform(df)
    
    # Intitalises a PCA to find a good number of principal components. 
    exploration_pca = PCA()
    exploration_pca.fit(df_scale)

    # creates a numpy array that stores the amount of variance explained by each principal component. 
    explained_variance = exploration_pca.explained_variance_ratio_

    # creates a numpy array with cumulative explained variance 
    cumulative_variance = np.cumsum(explained_variance)
    
    # finds fist cumulative explained variance value above 90%
    ninety_pca = [i for i in cumulative_variance if i >= 0.9][0]

    # initialises subplot plot
    plt.subplots(1, 2)
    plt.rc('font', size=6) 
    
    # plots the amount of variance explained by each consecutive
    plt.subplot(1, 2, 1)
    plt.plot(explained_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Proportion of Total Variance Explained')

    # plots the amount of variance explained when using different number of components.
    plt.subplot(1, 2, 2)
    plt.plot(cumulative_variance)
    plt.axhline(0.9, linestyle='--')
    plt.axvline(np.where(cumulative_variance == ninety_pca))
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Sum of Explained Variance')
    
    # reports the minimum number of components needed to explain 90% of the variance in the data
    print(np.where(cumulative_variance == ninety_pca)[0], "principal components needed to account for 90% of the", 
         "variance in the dataset")
    
    return plt
    

def pca_dim(df, scaler=None, n_components_=500):
    
    """
    Performs principal components analysis using a set number of components 
    
    Fits a dataframe to sklearn's PCA class with a selected number of principal components. 
    
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame on which PCA will be performed
    
    scaler: 'minmax' or 'standard' 
                a filepath to the location of the .csv file to be read in. 
                
    n_components_ : integer
                the number of components to be created


    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame of the reduced dimension data (will have the same number of columns 
                as the n_components used to fit the pca).
                
                
    Examples
    --------
    >>> pca_dim(df, scaler=minmax, n_components_=100):
    df
    
    See Also
    --------
    sklearn.PCA: function used to perform PCA 
    """
    
    # scales data based on scaler parameter 
    if scaler == 'minmax':
        df_scaled = MinMaxScaler().fit_transform(df)
    elif scaler == 'standard':
        df_scaled = StandardScaler().fit_transform(df)
    
    # intialises PCA and then fits and transforms data
    pca = PCA(n_components=n_components_)
    df_pca = pd.DataFrame(data=pca.fit_transform(df_scaled), index=df.index)
    
    return df_pca



def kmeans_opt(df, cluster_df, k_vals=list(range(2, 4))):
    """
    Fits a kmeans clustering on the data for a range a k values, then returns a dataframe with 
    clustering information.
    
    Takes in a dataframe of data to be clustered, df, a dataframe to record the clustering values, 
    cluster_df, and a range of k values to iterate over. Uses sklearn's KMeans to find clusters of data 
    based on distancs between data points.
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the data to perform clustering analysis on
    
    cluster_df: Pandas DataFrame
                a Pandas DataFrame of to report the clustering parameters and scores 
                (may already include previous clusterings iterations).
    
    k_vals : list
                a list of k values to pass to the KMeans algorithm.

    
    Returns
    -------
    cluster_df: Pandas DataFrame
                The updated cluster_df DataFrame, with new clustering information added
                
                
    Notes
    -----
    The passed in cluster_df DataFrame contains columns for parameters not relevant
    to KMeans. This is to allow for comparison between clusterings from different 
    algorithms. 
    
    Relevant columns for KMeans are:
    - model : pass in the instance of the model
    - silhouette score : pass in the calculated score f
    - k_val : pass in the value for k used 
    - inertia : pass in the inertia parameter
  
    
    See Also
    --------
    sklearn.KMeans : algorithm used to perform clustering analysis
    """
    
    # initialises lists for inertia and silhouette score
    inert = []
    silh = []
        
    # loop over the values for k 
    for k in k_vals:
        # intialises the KMeans algorithm with the current value for k
        km = KMeans(n_clusters=k, random_state=11)
        # fit the data and get the predicted cluster value for each datapoint 
        preds = km.fit_predict(df)

        # add instance of the model, silhouette score, inertia and k value to next row of cluster_df
        cluster_df.loc[len(cluster_df)] = {'model': km, 
                                           'silhouette_score': silhouette_score(df, preds),
                                           'k_val': int(k),
                                           'inertia': km.inertia_}
    
    return cluster_df

def dbscan_opt(df, cluster_df, epsi = np.arange(1.5,1.7,0.1), min_samples_=list(range(1, 3))):
    """
    Fits a DBScan clustering on the data for a range of epsilon and minimum samples values, then 
    returns a dataframe with clustering information.
    
    Takes in a dataframe of data to be clustered, df, a dataframe to record the clustering values, 
    cluster_df, and a range of epsilon values, epsi, and minimum sample values, min_sample_, to 
    iterate over. Uses sklearn's DBScan to find clusters of data based on density.
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the data to perform clustering analysis on
    
    cluster_df: Pandas DataFrame
                a Pandas DataFrame of to report the clustering parameters and scores 
                (may already include previous clusterings iterations).
    
    epsi : list
                a list of epsilon values to pass to the DBScan algorithm.
                
    min_samples_ : list
                a list of minimum sample values to pass to the DBScan algorithm.


    
    Returns
    -------
    cluster_df: Pandas DataFrame
                The updated cluster_df DataFrame, with new clustering information added
                
                
    Notes
    -----
    The passed in cluster_df DataFrame contains columns for parameters not relevant
    to DBScans. This is to allow for comparison between clusterings from different 
    algorithms. 
    
    Relevant columns for DBScan are:
    - model : pass in the instance of the model
    - silhouette score : the calculated score 
    - epsilon : the distance around a data point to look for other data points
    - min_samples_db : the minimum samples within the epsilon-defined area
    - number_noise_points : the number of points not assigned to a cluster
  
    
    See Also
    --------
    sklearn.DBScan : algorithm used to perform clustering analysis
    """
    
    # loop over minimum samples range
    for mini in min_samples_:
        # loop over epsilon values range (gets all combinations of minimum samples and epsilon
        for eps in epsi:
            # initialise the algorithm with the current values for epsilon and minimum samples
            db = DBSCAN(eps=eps, min_samples=mini)
            
            # predict the cluster labels for each data point
            labels = db.fit_predict(df)
            
            # isolate non-noise points
            non_noise = labels[labels != -1]
            
            # find number of clusters 
            num_cluster = len(np.unique(non_noise))
            
            # if there are enough clusters, calculate silhouette score 
            if num_cluster > 1:
                sil_score = silhouette_score(df, labels)
            else:
                sil_score = np.NaN
            
            # add current iteration data to the next row of the cluster_df dataframe 
            cluster_df.loc[len(cluster_df)] = {'model': db, 
                                               'epsilon': eps,
                                               'silhouette_score': sil_score, 
                                               'num_clusters': int(num_cluster),
                                              'min_samples_db': int(mini),
                                              'number_noise_points': len(labels) - len(non_noise)}
    return cluster_df


def agglo_opt(df, cluster_df, num_clusters_agglo=[3, 4]):
    """
    Fits a Hierarchical clustering on the data for a range clusters, then 
    returns a dataframe with clustering information.
    
    Takes in a dataframe of data to be clustered, df, a dataframe to record the clustering values, 
    cluster_df, and a range of epsilon values, epsi, and minimum sample values, min_sample_, to 
    iterate over. Uses sklearn's Agglomerative Clustering to find clusters of data based nearness.
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the data to perform clustering analysis on
    
    cluster_df: Pandas DataFrame
                a Pandas DataFrame of to report the clustering parameters and scores 
                (may already include previous clusterings iterations).
    
    num_clusters_agglo : list
                a list of number of clusters to try to fit to the data


    
    Returns
    -------
    cluster_df: Pandas DataFrame
                The updated cluster_df DataFrame, with new clustering information added
                
                
    Notes
    -----
    The passed in cluster_df DataFrame contains columns for parameters not relevant
    to AgglomerativeClustering. This is to allow for comparison between clusterings from different 
    algorithms. 
    
    Relevant columns for DBScan are:
    - model : pass in the instance of the model
    - silhouette score : the calculated score 
    - num_clusters : the number of clusters created
    - linkage : the distance metric used to calculate distances between clusters
  
    
    See Also
    --------
    sklearn.AgglomerativeClustering : algorithm used to perform clustering analysis
    """

    # create list of linkages to try
    linkage = ['ward', 'complete', 'average', 'single']
    
    # loop over linkages
    for link in linkage:
        # loop over number of clusters (gets all combinations of linkage and number of clusters)
        for num_clust in num_clusters_agglo:
            
            # intialise algorithm based on current parameters
            agglo = AgglomerativeClustering(n_clusters=num_clust, 
                                            linkage=link)

            # Fit and predict clustering labels
            labels = agglo.fit_predict(df)
            
            # add current iteration values to the cluster_df dataframe
            cluster_df.loc[len(cluster_df)] = {'model': agglo, 
                                               'silhouette_score': silhouette_score(df, labels),
                                               'num_clusters': int(num_clust),
                                               'linkage': link}

    return cluster_df



def clustering_adventure(df, k_vals=list(range(2, 4)), epsi = np.arange(1.5,1.7,0.1), 
                         min_samples_= list(range(1, 3)),
                        max_num_clusters_agglo = list(range(2, 4)),
                        max_num_clusters_gauss = 8):
    
    """
    Wrapper function to explore different clustering algorithms with varying parameters. 
    
    Takes in a dataframe of data to be clustered, df, and a selection of parameters for 
    different clustering algorithms. Calls functions to fit iterations of the 
    individual clusterings and stores all instances in a dataframe. Then creates plots 
    for each algorithm to compare instances. 
    
    Algorithms from sklearn:
        1) KMeans
        2) DBScan
        3) AgglomerativeClustering


    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of the data to perform clustering analysis on
    
    cluster_df: Pandas DataFrame
                a Pandas DataFrame of to report the clustering parameters and scores 
                (may already include previous clusterings iterations).
    
    k_vals : list
                a list of k values to pass to the KMeans algorithm.
    
    epsi : list
                a list of epsilon values to pass to the DBScan algorithm.
                
    min_samples_ : list
                a list of minimum sample values to pass to the DBScan algorithm.
    
    num_clusters_agglo : list
            a list of number of clusters to try to fit to the data


    
    Returns
    -------
    cluster_df: Pandas DataFrame
                The updated cluster_df DataFrame, with new clustering information added
    
    plt : MatPlotLib Figure Object 
                a object carrying the specifics of the plot created, so it can be altered or shown in 
                the notebook 
                
    Notes
    -----
    The passed in cluster_df DataFrame contains columns for parameters for all of the 
    algorithms, these are left null when not relevant. 
    
    
    cluster_df columns:
    - model : pass in the instance of the model
    - silhouette score : the calculated score 
    - k_val : pass in the value for k used (KMeans)
    - inertia : pass in the inertia parameter (KMeans)
    - epsilon : the distance around a data point to look for other data points (DBScan)
    - min_samples_db : the minimum samples within the epsilon-defined area (DBScan)
    - number_noise_points : the number of points not assigned to a cluster (DBScan)
    - num_clusters : the number of clusters created (AgglomerativeClustering)
    - linkage : the distance metric used to calculate distances (AgglomerativeClustering)
    
    See Also
    --------
    kmeans_opt : function to iterate over KMeans models
    dbscan_opt : function to iterate over DBScan models
    agglo_opt : function to iterate over AgglomerativeClustering Models
    """
    
    # start dataframe to keep track of the models and their values 
    cluster_df = pd.DataFrame(columns=['model', 'silhouette_score', 'num_clusters',
                                       'k_val', 'inertia', 
                                       'epsilon', 'min_samples_db', 'number_noise_points',
                                      'linkage'])
    
    # fit and predict data to kmeans, based on the passed in list of k_vals 
    cluster_df = kmeans_opt(df, cluster_df, k_vals)
    
    # fit and predict data to dbscan, based on the passed in list of k_vals 
    cluster_df = dbscan_opt(df, cluster_df, epsi, min_samples_)
    
    # fit and predict data to agglomerativeclustering, based on the passed in list of k_vals 
    cluster_df = agglo_opt(df, cluster_df, max_num_clusters_agglo)
    
    # create table with silhouette score for all combinations of epsilon and min_samples_db 
    db_mods = cluster_df[~cluster_df['epsilon'].isna()].pivot('epsilon', 
                                                              'min_samples_db', 'silhouette_score')
    
    # create table with silhouette score for all combinations of linkage and number of clusters
    agglo_mods = cluster_df[~cluster_df['linkage'].isna()].pivot('linkage', 
                                                                 'num_clusters', 'silhouette_score')

    
    # Create grid of plots for the iterations of the different algorithms
    plt.figure(figsize=(8,12))
    gridspec.GridSpec(3,2)
    
    # plot inertia for different values of k in KMeans
    plt.subplot2grid((3,2), (0,0), rowspan=1, colspan=1)
    sns.lineplot(data=cluster_df[~cluster_df['k_val'].isna()], x='k_val', y='inertia')
    plt.title("KMeans Inertia")
    plt.ylabel("Inertia")
    plt.xlabel("Number of Clusters")
    plt.xticks(k_vals)

    # plot silhouette score for different values of k in KMeans
    plt.subplot2grid((3,2), (0,1), rowspan=1, colspan=1)
    sns.lineplot(data=cluster_df[~cluster_df['k_val'].isna()], x='k_val', y='silhouette_score')
    plt.title("KMeans Silhouette Score")
    plt.ylabel("Silhouette Score")
    plt.xlabel("Number of Clusters")
    plt.xticks(k_vals)
    
    # plot heatmap of silhouette score for epsilon and minimum samples combinations
    plt.subplot2grid((3,2), (1,0), rowspan=1, colspan=2)
    sns.heatmap(db_mods)
    plt.ylabel("Epsilon")
    plt.xlabel("Minimum Samples for Core Point")
    plt.title("DBScan Parameters by Silhouette Score")

    # plot heatmap of silhouette score for linkage and number of clusters combinations
    plt.subplot2grid((3,2), (2,0), rowspan=1, colspan=2)
    sns.heatmap(agglo_mods)
    plt.ylabel("Linkage")
    plt.xlabel("Number of Clusters")
    plt.title("Hierarchical Parameters by Silhouette Score")

    return cluster_df, plt



def cluster_labels(df, cluster_df, kmeans=True, dbscan=True, agglo=True, 
                   opt_k=3, opt_eps=1.5, opt_min_samples=2.0, 
                   opt_linkage='ward', opt_num_agglo_clusters=2.0):
    

    """
    Adds clustering labels for selected algorithms to dataframe. 
    
    Takes in chosen optimal parameters for selected algorithms and adds clustering labels for 
    those specific clustering instances to the dataframe. 
    
    Parameters
    ----------
    df : Pandas DataFrame
            a Pandas DataFrame of the data 
    
    cluster_df : Pandas DataFrame
                a Pandas DataFrame of to report the clustering parameters and scores 
                (may already include previous clusterings iterations).
    
    kmeans : boolean
                Whether or not to include KMeans labels in dataframe
                
    dbscan : boolean
                Whether or not to include DBScan labels in dataframe
    
    agglo : boolean
                Whether or not to include AgglomerativeClustering labels in dataframe
                
    opt_k : integer
                optimal value for k for KMeans
                
    opt_eps : float 
                optimal value for epsilon for DBSCan
                
    opt_min_samples : float
                optimal value for minimum samples for DBScan
                
    opt_linkage : string
                optimal linkage for AgglomerativeClustering
                
    opt_num_agglo_clusters : 
                optimal number of clusters for AgglomerativeClustering
               

    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame with the additional cluster labels columns

    
    See Also
    --------
    clustering_adventure : function used to iterate over models and model parameters.
    """

    # create a copy of the passed in dataset to alter
    df_labelled = df.copy()
    
    # check to see if KMeans should be added
    if (kmeans == True):
        # gets the optimal model from the cluster_df dataframe, based on the parameter location
        opt_km = cluster_df[cluster_df['k_val'] == opt_k]['model'].values[0]
        # fit data to get labels
        df_labelled['kmeans_labels'] = opt_km.predict(df)
        
    # check to see if DBScan should be added
    if (dbscan == True):
        # gets the optimal model from the cluster_df dataframe, based on the parameter location
        opt_db = cluster_df[(cluster_df['epsilon'] == opt_eps) & 
                            (cluster_df['min_samples_db'] == opt_min_samples)]['model'].values[0]
        # fit data to get labels 
        df_labelled['db_labels'] = opt_db.fit_predict(df)
    
    # check to see if AgglomerativeClustering should be added
    if (agglo == True):
        # gets the optimal model from the cluster_df dataframe, based on the parameter location
        opt_agglo = cluster_df[(cluster_df['linkage'] == opt_linkage) & 
                            (cluster_df['num_clusters'] == opt_num_agglo_clusters)]['model'].values[0]
        # fit data to get labels 
        df_labelled['agglo_labels'] = opt_agglo.fit_predict(df)
    
    return df_labelled


def tsne_vis_unlabelled(df, num_components_=3):
    
    """
    Performs a tSNE dimensionality reduction based on the passed in data and number of components,
    and creates pairwise plots of the data.
    
    
    Parameters
    ----------
    df: Pandas DataFrame
                a Pandas DataFrame of data to be reduced 
                
    num_components : int
                the number of dimensions to reduce to 

    Returns
    -------
    plt : MatPlotLib Figure Object 
                a object carrying the specifics of the plot created, so it can be altered or shown in 
                the notebook 
   
    See Also
    --------
    sklearn.TSNE : function used to iterate over models and model parameters.
    """
    
    # intialise tSNE 
    tSNE = TSNE(n_components=3)
    
    # fit and transform data to 
    tSNE_data = tSNE.fit_transform(df)

    # create plt instance and pairwise plot of the distributions of the data by dimension
    plt.figure()
    sns.pairplot(pd.DataFrame(tSNE_data))
    
    return plt


def tsne_vis(df_labelled, num_components=3):
    
    """
    Performs a tSNE dimensionality reduction based on the passed in data and number of components,
    and creates pairwise plots of the data labeled by different clusterings
    
    
    Parameters
    ----------
    df_labelled: Pandas DataFrame
                a Pandas DataFrame of data to be reduced 
                
    num_components : int
                the number of dimensions to reduce to 

    Returns
    -------
    plt : MatPlotLib Figure Object 
                a object carrying the specifics of the plot created, so it can be altered or shown in 
                the notebook 
   
    See Also
    --------
    sklearn.TSNE 
    """

    # initalise tSNE and fit data (except clustering labels)
    tSNE = TSNE(n_components=num_components)
    tSNE_data = tSNE.fit_transform(df_labelled.drop(columns=['kmeans_labels', 
                                                            'agglo_labels']))
    
    # create dataframe of tSNE reduced data with clustering labels
    tSNE_df = pd.DataFrame(tSNE_data, 
                           columns=[f'tSNE_{i+1}' for i in range(tSNE_data.shape[1])])

    # add all the algorithm names in the dataset to a list named algos 
    algos = []
    
    if (np.isin(df_labelled.columns.values, 'kmeans_labels').any()):
        algos.append('kmeans_labels')
        tSNE_df['kmeans_labels'] = df_labelled['kmeans_labels'].values
    if (np.isin(df_labelled.columns.values, 'dbscan_labels').any()):
        algos.append('dbscan_labels')
        tSNE_df['db_labels'] = df_labelled['db_labels'].values 
    if (np.isin(df_labelled.columns.values, 'agglo_labels').any()):
        algos.append('agglo_labels')
        tSNE_df['agglo_labels'] = df_labelled['agglo_labels'].values
        

    num_algos = len(algos)
    
    # return simple pairwise plot with single clustering if there is only one
    if (num_algos == 1):
        # create plot and color data points by their cluster
        plt.figure()
        sns.pairplot(tSNE_df, hue=algos[0], plot_kws={'alpha':0.5})
        return plt
    
    # create a list of the labels names 
    labels = np.array(algos.copy())
    
    plots = []
    
    # iterate over algorithms and create pairwise plots for the tSNE data with clusters
    # all in the same colors. 
    for i in range(num_algos):
        algo = algos[i]
        others = np.delete(labels, np.where(labels == algo))
        plt.figure()
        sns.pairplot(tSNE_df.drop(columns=others), hue=algo, plot_kws={'alpha':0.5})

        # add appropriate titles to plots 
        if (algo == 'kmeans_labels'):
            plt.title("KMeans")
        elif (algo == 'db_labels'):
            plt.title("DBScan")
        elif (algo == 'agglo_labels'):
            plt.title("Hierarchical")
            
        plots.append(plt)
        
    return plots



def rec_data_readin():
    """
    Reads in the three final cleaned and reduced datasets to be used in recommendation systems.
    
    The song information dataset (read in as sl_names) is formatted to produce a nicer output
    for our recommendations. The track column is also dropped because it is not useful in this
    iteration of the project (though this could easily be undone for future work).
    
    
    Parameters
    ----------
    No parameters are directly passed in, but the filepaths for the locations of the csv files 
    are hard coded into the read-in functions

    Returns
    -------
    sl_red : Pandas DataFrame
                a Pandas DataFrame of the dimensionality reduced song data 
                
    sl_names : Pandas DataFrame
                a Pandas DataFrame including the title, artist name and release for every song in 
                sl_red (for recommendation output) formatted nicely
                
    tp : Pandas DataFrame
                a Pandas DataFrame containing the taste profile for specific users.

   
    See Also
    --------
    ds_utils_capstone.read_csv_pd : function used to read in csv files
    """
    
    print("Song Data Read In")
    # read in song data, with song as index
    sl_red = ds_utils_capstone.read_csv_pd('data/sub_lmd_pca.csv', index=True)
    
    print("\nSong Name Information Read In")
    # read in song name data, with song as index
    sl_names = ds_utils_capstone.read_csv_pd('data/sub_lmd_names.csv', index=True)

    # removing track id for this iteration of the project
    sl_names = sl_names.drop(columns='track_id')
    
        
    print("\nTaste Profile Read In")
    # read in taste profile, with no index
    tp = ds_utils_capstone.read_csv_pd('data/sub_taste.csv', index=False)
    
    return sl_red, sl_names, tp


def scaling_mmb(tp, scale_type='allinone'):
    

    taste_ones = tp[tp['count'] == 1.0].copy()
    taste_ones['rating'] = taste_ones['count']
    
    taste_repeat = tp[tp['count'] > 1].copy()

    if (scale_type == 'allinone'):
        
        count_scaled = MinMaxScaler().fit_transform(stats.boxcox(taste_repeat['count'])[0].reshape(-1,1))
        
        for i in range(len(count_scaled)):
            if count_scaled[i] > 0.75:
                count_scaled[i] = 5
            elif count_scaled[i] > 0.5:
                count_scaled[i] = 4
            elif count_scaled[i] > 0.25:
                count_scaled[i] = 3
            else:
                count_scaled[i] = 2
                

        taste_repeat['rating'] = count_scaled

    elif (scale_type == 'userbased'):
        
        unique_user = taste_repeat['user'].unique()

        taste_repeat_norm = pd.DataFrame(columns=['user', 'song', 'count'])

        for user in unique_user:
            user_df =  taste_repeat[taste_repeat['user'] == user].copy()
            user_counts = user_df['count']

            if np.all(user_counts.values == user_counts.values[0]):
                count_scaled = np.full((len(user_counts.values), 1), 3.0)
            else:
                count_scaled = MinMaxScaler().fit_transform(stats.boxcox(user_counts.values)[0].reshape(-1,1))

            user_df['count'] = count_scaled 

            taste_repeat_norm = pd.concat([taste_repeat_norm, user_df])


        taste_repeat_norm['rating'] = pd.cut(taste_repeat_norm['count'],
                                           bins=[-np.inf, 0.25, 0.5, 0.75, np.inf],
                                           labels=[2.0,3.0,4.0,5.0])
        
        taste_repeat = taste_repeat_norm
        
    else:
        return "Please pass an appropriate rating format."
        

    taste_scaled = pd.concat([taste_repeat, taste_ones]).drop(columns=['count'])
    
    print(taste_scaled['rating'].value_counts())

    return taste_scaled


def dist_considerations(sr):
    
    ss = StandardScaler()
    mm = MinMaxScaler()
    
    log = np.log(sr.values).reshape(-1, 1)
    boxcox = stats.boxcox(sr.values)[0].reshape(-1,1)
    
    gridspec.GridSpec(2,2)
    
    plt.subplot2grid((2,2), (0,0))
    sns.histplot(ss.fit_transform(log))
    plt.title("StandardScaler and Log Transform")
    plt.legend('')
    
    plt.subplot2grid((2,2), (0,1))
    sns.histplot(ss.fit_transform(boxcox))
    plt.title("StandardScaler and BoxCox Transform")
    plt.legend('')
    
    plt.subplot2grid((2,2), (1,0))
    sns.histplot(mm.fit_transform(log))
    plt.title("MinMax and Log Transform")
    plt.legend('')
    
    plt.subplot2grid((2,2), (1,1))
    sns.histplot(mm.fit_transform(boxcox))
    plt.title("MinMax and BoxCox Transform")
    plt.legend('')
    
    return plt


def track_mp3_to_wav(directory, set_tracks):
    """
    """
    
    print(directory)
    
    if not os.path.exists(directory):
        raise Exception("Directory %s doesn't exist, are you sure this is what you're looking for?" % directory)
    
    # Here we are initialising the location for base of our filepath for the new .wav files 
    folder_base = 'data/sect_wavs/'
    
    # Set size to start 
    print("Total number of tracks to find:", len(set_tracks))
    
    # We are going to iterate over all of the folders within the audio folder
    for root, dirnames, filenames in os.walk(directory): 
        # Only in cases where there are files in the directory 
        if filenames != []:
            # We loop over the files in the directory (usually one but there are some multiple cases)
            for filename in filenames:
                
                # Create a variable to store the name of the current mp3 track
                track_name = filename[:-4]
                
                # Check if the current mp3 track name is in our subset 
                if track_name in set_tracks:
                    
                    # Initialising the location of the mp3 file for the pydub .from_mp3 function 
                    src = root + '/' + filename
                    # Using the location of the mp3 file to create a new location within the sect_wav folder  
                    dst = folder_base + root[-5:] + '/' + track_name + '.wav'
                    # read in the mp3 file
                    tune = AudioSegment.from_mp3(src)
                    
                    # export the file as a wav file 
                    tune.export(dst, format="wav")
                    
                    # As we have seen the track, we remove it from our set (this lightens computational load)
                    set_tracks.remove(track_name)
                    

                    # print the number of tracks not found 
                    print("Number of tracks left to find:", len(set_tracks))
    
    # if there are unfound tracks, print their names 
    if (len(set_tracks) != 0):
        print("Unfound Tracks:", set_tracks)
        
        
def voc_sep(filepath, graph=False):
    y, sr = librosa.load(path=filepath)
    
    # computing spectogram magnitude and phase(angle) (will only use the magnitude) after computing discrete fourier 
    # transforms over overlapping windows (overlap by 1/4 in this case)
    S_full, phase = librosa.magphase(librosa.stft(y))

    # get a 5 second slice of the spectrum 
    idx = slice(*librosa.time_to_frames([2, 7], sr=sr))

    # # We'll compare frames using cosine similarity, and aggregate similar frames
    # by taking their (per-frequency) median value.
    #
    # To avoid being biased by local continuity, we constrain similar frames to be
    # separated by at least 2 seconds.
    #
    # This suppresses sparse/non-repetetitive deviations from the average spectrum,
    # and works well to discard vocal elements.

    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))

    # The output of the filter shouldn't be greater than the input
    # if we assume signals are additive.  Taking the pointwise minimium
    # with the input spectrum forces this.
    S_filter = np.minimum(S_full, S_filter)


    # We can also use a margin to reduce bleed between the vocals and instrumentation masks.
    # Note: the margins need not be equal for foreground and background separation
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    # Once we have the masks, simply multiply them with the input spectrum
    # to separate the components

    S_foreground = mask_v * S_full
    S_background = mask_i * S_full
    
    
    if graph == True:
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        librosa.display.specshow(librosa.amplitude_to_db(S_full[:, idx], ref=np.max),
                                 y_axis='log', sr=sr)
        plt.title('Full Spectrogram')
        plt.colorbar()

        plt.subplot(3, 1, 2)
        librosa.display.specshow(librosa.amplitude_to_db(S_background[:, idx], ref=np.max),
                                 y_axis='log', sr=sr)
        plt.title('Background (Instrumentation)')
        plt.colorbar()
        plt.subplot(3, 1, 3)
        librosa.display.specshow(librosa.amplitude_to_db(S_foreground[:, idx], ref=np.max),
                                 y_axis='log', x_axis='time', sr=sr)
        plt.title('Foreground (Vocals)')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('images/vocal_sep.png')
        plt.show()
    
    return S_foreground 


def voc_dict_maker(directory, set_tracks):
    
    if not os.path.exists(directory):
        raise Exception("Directory %s doesn't exist, are you sure this is what you're looking for?" % directory)
    
    vocal_isol_pd = pd.DataFrame(columns=['track_name', 'foreground_matrix'])
    
    set_tracks_tracker = set_tracks.copy()
    
    # Set size to start 
    print("Total number of tracks to find:", len(set_tracks))
    
    # We are going to iterate over all of the folders within the audio folder
    for root, dirnames, filenames in os.walk(directory): 
        # Only in cases where there are files in the directory 
        if filenames != []:
            
            # We loop over the files in the directory (usually one but there are some multiple cases)
            for filename in filenames:
                
                # Create a variable to store the name of the track (so we can match it to the data)
                track_name = filename[:-4]
                
                # Check if the current mp3 track name is in our subset 
                if track_name in set_tracks_tracker:
                    
                    vocal_isol_pd = pd.concat([vocal_isol_pd, pd.DataFrame({'track_name': [track_name], 
                                                                        'foreground_matrix':
                                                                            [voc_sep(root + '/' + filename)]})])
                    
                    # As we have seen the track, we remove it from our set (this lightens computational load)
                    set_tracks_tracker.remove(track_name)

                    # print the number of tracks not found 
                    print("Number of tracks left to find:", len(set_tracks_tracker))

    # if there are unfound tracks, print their names 
    if (len(set_tracks) != 0):
        print("Unfound Tracks:", set_tracks_tracker)
        
    return  vocal_isol_pd
    
