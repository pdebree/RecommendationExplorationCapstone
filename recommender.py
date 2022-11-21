import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

import ds_utils_capstone
import capstone_utils
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset
from surprise.reader import Reader
from surprise.prediction_algorithms.matrix_factorization import SVD as FunkSVD
from surprise.model_selection import LeaveOneOut

import random

from surprise import accuracy
from surprise.model_selection import train_test_split as train_test_splitSurprise

from surprise.model_selection import GridSearchCV as GridSearchCVSurprise
from surprise.model_selection import cross_validate

from scipy import stats

from scipy.stats import rankdata
from scipy.stats import kendalltau

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import math


def taste_profile_evaluation(taste_profile_with_predictions, rec_model, df=None,
                            tau_eval=True, prediction_column = 'prediction'):
    """
    Takes in a dataframe that holds evaluations for different recommendation system iterations and 
    adds the evaluation metrics
    
    If the dataframe is not passed in - it is initiated in the call of the function (this may be used for the
    first instance of evaluation)
    """
    
    if (df is None):
        df = pd.DataFrame(columns=['rec_model', 'rmse', 'mae', 'kendalltau'])
        
    
    if (np.isin(taste_profile_with_predictions.columns.values, prediction_column).any() == False):
        return print("No appropriate prediction column passed in. \nThe columns in the passed in prediction table are:\n\t",
                     taste_profile_with_predictions.columns.values)
        
    
    counts = taste_profile_with_predictions['count']
    true = taste_profile_with_predictions['scaled_score']
    predicted = taste_profile_with_predictions[prediction_column]

           
    tau = np.NaN
           
           
    if (tau_eval == True):
        # need to do kendalls tau by user - we will report an average tau 
        unique_user = taste_profile_with_predictions['user'].unique()

        tau_sum = 0 
        successful_kendall = 0 

        for user in unique_user:
            user_data = taste_profile_with_predictions[taste_profile_with_predictions['user'] == user]
            counts_ranked = rankdata(user_data['count'])
            predictions_ranked = rankdata(user_data[prediction_column])

            # kendall's tau (tau-b accounts for ties )
            tau, p_value = kendalltau(counts_ranked, predictions_ranked)

            if (math.isnan(tau) == False):
                successful_kendall += 1
                tau_sum += tau 

        if successful_kendall > 0:
            tau = tau_sum / successful_kendall 
        else:
            print("Unable to perform Kendall's Tau Analysis - too many ties")

    
    
    df.loc[len(df)] = {'rec_model': rec_model,
                       'rmse': mean_squared_error(true, predicted, squared=False), # pass in false to get root
                       'mae': mean_absolute_error(true, predicted), 
                       'kendalltau':tau}

    return df 



def rec_top_songs(taste_profile, song_info, by='total_plays', number_recs=9):
    """
    Selects and reports most popular songs in the passed in taste_profile.
    
    Calculates the most popular songs and recommends them to a new users by adding the actual song 
    informaton from the song_info dataframe.
    
    Creates, and returns, the DataFrame and prints the shape of the created DataFrame and 
    whether duplicated or missing values were found (usedin no_duplicates_missing). This functionality is 
    useful when reading in known, cleaned files to ensure they take the expected form.
    
    Parameters
    ----------
    taste_profile: Pandas DataFrame
                Pandas DataFrame with the play counts for specific user and song combinations
                
    song_info : Pandas DataFrame
                Pandas DataFrame with the song information (title, artist name and release)
                
    by : string
                A string to depict the metric to define top songs ('total_plays' or 'number_listeners')
                
    number_recs : int
                The number of recommendations to return to a user
    
    
    Returns
    -------
    df: Pandas DataFrame
                a Pandas DataFrame of nine most popular songs, based on the popularity metric

    
    Note
    ----
    'total_plays' bases popularity on the total number of times a song was listen to (all user listens)
    'number_listeners' bases popularity on the number of distinct users who have listened to a song
    
    """
    
    if (by == 'total_plays'):
        # count total listens by song
        total_song_counts = taste_profile.groupby(['song'])['count'].sum().to_frame().sort_values(by='count', ascending=False)
        # return songs with the highest total_song_counts
        return song_info[song_info.index.isin(total_song_counts.head(number_recs).index.values)]
    
    elif (by == 'number_listeners'):
        # count totals by number of unique listeners
        listeners_counts = taste_profile.groupby('song')['user'].count().to_frame().sort_values(by='user', ascending=False)
        # return songs with the highest listener_counts
        return song_info[song_info.index.isin(listeners_counts.head(number_recs).index.values)]
    
    else:
        print("Please enter appropriate popularity measure:\n\t\"total_plays\" or \"number_listeners\"")


class ContentRecommender:
    """
    This class acts as a content based recommendation system, by building a similarity matrix on which 
    functions can be called to generate recommendations for users who are known to enjoy a specific small
    set of songs. Though not a complicated recommendation, it returns to the user songs that have close 
    cosine similarities to those they are known to like. 
    
    Parameters
    ----------
    taste_profile: Pandas DataFrame
                a Pandas DataFrame of a song taste profile - must include a count column - to be initialised
                as self.taste_profile
                
    song_data : Pandas DataFrame
                a Pandas DataFrame of specific song data for each song found within the taste profile - to 
                be initialised as self.song_data
    
    song_info : Pandas DataFrame
                a Pandas DataFrame of the song title, artist name and release for recommendation output - to 
                be initialised as self.song_info
    
    rec_number : int
                The number of recommendations set to be given to a user - to be intialised into self.rec_number
    
    number_of_songs_to_consider : int
                Number of songs to consider within the taste profile (if these can be found - to be intialised 
                into self.number_songs_to_consider

    Notes
    -----
    The initialisation also creates a cosine similarity matrix using the class cosine_sim_matrix function, 
    which takes in the song data. 
    
    The assumption that a user has enjoyed a song is built on heavily here, as if there are only few or one song a
    user has listened to, there is no option in skewing towards more favored (listened to) songs. 
    """
    
    
    def __init__(self, taste_profile, song_data, song_info, rec_number=12, number_songs_to_consider=3):

        self.taste_profile = taste_profile
        self.song_data = song_data
        self.song_info = song_info
        self.rec_number = rec_number
        self.number_songs_to_consider = number_songs_to_consider
        self.similarity = self.cosine_sim_matrix()

        
    def cosine_sim_matrix(self, dense_output_=False):
        """
        Creates a cosine similarity matrix based on self.song_data 
        
        Uses sklearn pairwise metrics function cosine_similarity to calculate a matrix of 
        similarity between every song in the dataset. This can then be used to find the most 
        similar songs. 


        Paratmeters
        -----------
        dense_output_ = Boolean
                    boolean of whether to create a sparse or dense cosine similarity matrices

        Returns
        -------
        df : Pandas DataFrame
                A Pandas DataFrame of the sparse similiarity matrix with labeled columns and 
                indices for the song combinations. 


        See Also
        -------
        sklearn.metrics.pairwise.cosine_similarity : 
        """
    
    
        # create similiarity matrix - need to transpose so that 
        sim_matrix = cosine_similarity(self.song_data, dense_output=dense_output_)

        # get song names from pandas dataframe
        indices = self.song_data.index.values

        # create dataframe of similarities with indices and columns labelled.
        return pd.DataFrame(data=sim_matrix, index=indices, columns=indices)


    def content_recs(self, song_index, number_to_rec = None):
        """
        Creates a dataframe of song recommendations based on a passed in song. 
        
        Uses the self.similiarity matrix to find songs that are most similar to the one 
        passed in. number_to_rec takes in a number of recommendations to return, but if 
        None are called this is based on the number of songs need to be recommended if 
        multiple songs will be used in the recommendation. 
        

        Paratmeters
        -----------
        song_index : string
                    the string of the specific song that recommendations will be based on.
                    
        number_to_rec : int or None
                    the number of songs to recommend based on the passed in song

        Returns
        -------
        df : Pandas DataFrame
                A Pandas DataFrame of the song information for the given songs  

        """
    
        # checks whether a specific number of recommendations is passed in 
        if (number_to_rec is None):
            number_to_rec = self.rec_number//self.number_songs_to_consider
            

        # get the n songs with the largest similiarities 
        # (but will also get a 1 for the same song vector, so need to ignore it)
        recs = self.similarity.loc[song_index].sort_values(ascending=False).head(number_to_rec).index.values[1:]
        
        # create a dataframe of the information about the songs to be recommended 
        rec_details = self.song_info[self.song_info.index.isin(recs)]

        return rec_details

    
    def recommend_user(self, user = None):
        """
        Creates an output of songs recommendations based on the most enjoyed songs by a specific 
        user. 
        
        Randomises a user, if one is not directly passed and uses self.taste_profile to find songs 
        most enjoyed by the user by calling the self.content_rec function for each song. This returns 
        a number of recommendations (defined by the floor of dividing self.rec_number by 
        self.number_songs_to_consider). If the user does not have as many songs as the set 
        number of songs to consider, the number of recommendations from each call to the self.contant_rec
        function is defined by the self.rec_number divided by the number of songs there are for the specific
        user. 
        
        These recommendations are concatenated into a single dataframe and returned to the system user.

        Paratmeters
        -----------
        user : None
                    alphanumeric code for specific user to be considered, if None passed a user is 
                    randomly selected within the function.


        Returns
        -------
        df : Pandas DataFrame
                A Pandas DataFrame of the song recommendations


        See Also
        -------
        self.content_recs : function used to get similar songs (based on cosine similarity) for recommendation
        
        """
        
        # select random user if one is not passed in
        if (user is None):
            user = self.taste_profile['user'].values[random.randint(0, len(self.taste_profile['user']) - 1)]

        # get the user's taste profile
        user_songs = self.taste_profile[self.taste_profile['user'] == user]
        
        # gets number of songs to consider set in initialisation
        num_songs_to_consider = self.number_songs_to_consider
        
        # if user appears less times than the limit, use all songs they have listened to 
        if num_songs_to_consider > len(user_songs):
            num_songs_to_consider = len(user_songs)
            
        # select specific songs to consider, based on the highest listen count
        user_profile = self.taste_profile[self.taste_profile['user'] == user].nlargest(num_songs_to_consider,'count')
        mult_songs_recs = pd.DataFrame()
        
        # for songs to use in recommendation, get recommendations from content_recs
        for song in user_profile['song']:
                mult_songs_recs = pd.concat([mult_songs_recs, 
                                             self.content_recs(song, self.rec_number // num_songs_to_consider)], 
                                            axis=0)
        
        return mult_songs_recs
        
    
    def similar_songs(self, song=None, number_to_recommend=6):
        """
        Simple song recommendation based on passing song into dataset, instead of using user's taste profile.
        
        If no song is passed in, one is randomly selected from the taste profile (for ease of system use).
        
        Paratmeters
        -----------
        song : None
                    alphanumeric code for specific song to be considered, if None passed a song is 
                    randomly selected within the function.
                    
        number_to_recommend : int
                    The number of songs to recommend based on the song


        Returns
        -------
        df : Pandas DataFrame
                A Pandas DataFrame of the song recommendations


        See Also
        -------
        self.content_recs : function used to get similar songs (based on cosine similarity) for recommendation
        """
        
        # randomly select song, if none is passed in 
        if (song == None):
            song = self.taste_profile['song'].values[random.randint(0, len(self.taste_profile['song']) - 1)]
            
        # return recommendations based on chosen song as a dataframe
        return self.content_recs(song, number_to_rec = number_to_recommend)
        
    
class CollaborativeSongRecommender:
    """
    This class acts as a user-similarity collaborative filtering recommendation system. 
    
    By building a utility matrix (of user-song combinations) the similarity between users can be
    found, if there is a large enough overlap. 
    

    Parameters
    ----------
    taste_profile: Pandas DataFrame
                a Pandas DataFrame of a song taste profile - must include a count column - to be initialised
                as self.taste_profile
                
    song_data : Pandas DataFrame
                a Pandas DataFrame of specific song data for each song found within the taste profile - to 
                be initialised as self.song_data
    
    song_info : Pandas DataFrame
                a Pandas DataFrame of the song title, artist name and release for recommendation output - to 
                be initialised as self.song_info
    
    rec_number : int
                The number of recommendations set to be given to a user - to be intialised into self.rec_number
                
    scale_by : string
                A string noting whether the scaling should be performed all together or by individual user, this
                is passed into the capstone_utils.scaling_mmb file 
    
    user_matrix_df : Pandas DataFrame
                A Pandas DataFrame of a user-latent feature matrix obtained by FUNK SVD
    
    song_overlap : int
                Number of songs to that two users must have in common to consider the ratings of the other (if 
                this is too low irrelevant users may skew the predicted rating).
                
    needed_similar_rating : int
                Number of similar users who have rated a specific song, for a prediction to be made
                

    Notes
    -----
    The initialisation also creates a utility matrix using the class's utility_matrix function. This will be used 
    to easily compare the ratings of different users. 
    
    Though a rating system is not necessary for the structure of this recommendation system, it works as a normalisation
    that allows different users to be compared (where differences in listen counts could create skews).
    
    See Also
    --------
    SurpriseDecompositionRecommender : a class that includes the decomposition necessary for using a user_matrix_df
    """
    
    def __init__(self, taste_profile, song_info, rec_number=10, scale_by = 'allinone',
                user_matrix_df = pd.DataFrame(), song_overlap=3, needed_similar_rating=2):
        
        self.taste_profile = taste_profile
        self.scaled_taste_profile = capstone_utils.scaling_mmb(self.taste_profile, scale_type=scale_by)
        
        self.song_info = song_info
        self.rec_number = rec_number
        self.utility = self.utility_matrix()
        self.user_matrix_df = user_matrix_df
            
        self.song_overlap = song_overlap
        self.needed_similar_rating = needed_similar_rating 
        
    
    def utility_matrix(self, scale=True):
        """
        Explicitly creates a Utility Matrix 
        
        Uses the class' scaled_taste_profile to create a Utility Matrix. Note the matrix created 
        is not based on the user count but rather the listen count 
        
        and then returns a matrix with every user in a row and every song in a column.
        The values of the matrix are then filled with the number of listen counts the player has 
        for each song. (This does result in a vary sparse matrix)

        In this case we are going to use the scaled taste profile
        
        word score, because we want to maintain the 
        functionality of having access to both ratings and song counts as enjoyment idicators.
        

        
        Paratmeters
        -----------
        scale: Boolean 
                    A boolean of whether the rated or count data should be used in creating the 
                    utility matrix
        
        Returns
        -------
        R : Pandas DataFrame
                A Pandas DataFrame of utility matrix (passed into the self.utility matrix attribute)
        """
        
        assert isinstance(scale, bool), "Must pass a boolean value to \'scale\'" 
        
        # if scaling should not be used, use original rated data (counts)
        if (scale == False):
            taste_profile = self.taste_profile
        else:
            taste_profile = self.scaled_taste_profile
            
        print("\nStarting Utility Matrix Formation")
        
        # finding unique users and the number 
        uni_users = taste_profile['user'].unique()
        num_users = len(uni_users)

        # finding unique songs and the number
        uni_songs = taste_profile['song'].unique()
        num_songs = len(uni_songs)

        # Initiate DataFrame null matrix with number of users and songs for dimensions
        R_numpy = data=np.full((num_users, num_songs), np.nan)
        R = pd.DataFrame(data=R_numpy, index= uni_users, columns=uni_songs)                    

        # iterate over taste profile to create utility matrix
        for triplet in taste_profile.itertuples(): 
            user = triplet[1]
            song = triplet[2]
            score = triplet[3]
            R.loc[user, song] = score

        # return size of created utility matrix
        print(f'Created Utility Matrix with:\n\t {num_users} Unique',
              f'Users\n\t{num_songs} Unique Songs')

        return R

    
    def similar_users_to_user(self, curr_user = None):
        """
        Finds similar users and their cosine similarity, based on overlap passed into class initialisation
        
        Uses self.song_overlap to define a number of songs that is considered high enough for two users to be 
        seen as similar. User combinations between the current user (who is randomised if not passed in) and 
        others are then tested based on cosine similiarity between one or two metrics. All similiarities are
        found based on the similarity between ratings. If a user_matrix from a singular value decomposition is
        passed into the class initiatialisation latent feature similarity can also be calculated using 
        cosine similarity. 
           
        Paratmeters
        -----------
        curr_user : string
                    A unique identifier for a specific user, for whom similar users will be found

        Returns
        -------
        user_similarity_overlap : Pandas Series
                    A Pandas Series of users and their cosine similiarity to the current user based on the 
                    similarity in ratings 
                        
        
        user_similarity_latent : Pandas Series 
                    A Pandas Series of users and their cosine similarity to the current user based on the 
                    similiarity to latent features from a user matrix derived from a SVD 

        Note
        ----
        The user matrix needed for comparing latent features can be created using an SVD in the 
        SurpriseDecompositionRecommender class. Latent features could be calculated regardless of overlap but
        ensuring that the overlap threshold is met allows us to be sure that only users who have behaved 
        similarly (i.e. listened to the same songs) are compared.
        
         
        See Also 
        --------
        SurpriseDecompositionRecommender: A Class for recommending based on a Singular Value Decomposition
        """
        
        # If no user is explicitly passed, one is randomly selected  
        if (curr_user == None):
            curr_user = self.taste_profile['user'].values[random.randint(0, len(self.taste_profile['user']) - 1)]
        
       
        # create series for overlaps 
        user_similarity_overlap = pd.Series(dtype='float64')
        user_similarity_latent = pd.Series(dtype='float64')
        
        # find the current user's (to be compated to) ratings and rated columns (songs)
        curr_user_rated = ~self.utility.loc[curr_user,:].isna()
        curr_user_columns = self.utility.loc[curr_user].dropna().index.values
        

        # for each user (not the one we are considering)
        for other_user in self.utility.loc[self.utility.index != curr_user].index.values:
            
            # find section of songs that both have listened to 
            both_rated = curr_user_rated & ~self.utility.loc[other_user,:].isna()
            both_rated_columns = both_rated[both_rated].index.values
            
            # if the overlap between what they have both rated reaches the threshold 
            if (both_rated.sum() > self.song_overlap):

                # create song based similarity matrix
                # find the current users ratings for the overlapping songs
                curr_user_ratings = self.utility.loc[curr_user, both_rated].values.reshape(1, -1)

                # find the other users' ratings for the overlapping songs
                other_user_ratings = self.utility.loc[other_user, both_rated].values.reshape(1, -1)
              
                # add other user and similarity to overlap series 
                user_similarity_overlap[other_user] = cosine_similarity(curr_user_ratings, other_user_ratings)
                
                # add in latent features similarity (if available)
                if (len(self.user_matrix_df) > 0):
                    user_similarity_latent[other_user] = cosine_similarity(
                        self.user_matrix_df.loc[curr_user].values.reshape(1,-1), 
                        self.user_matrix_df.loc[other_user].values.reshape(1, -1))
        
        # return user similarities 
        return user_similarity_overlap, user_similarity_latent

    
    def user_rating_prediction(self, curr_user = None, song = None):
        """
        Predicts a user's rating based on weighing the predictions of other users based on their similarity
        
        
        Creates a predicted rating for a song and user combination by using the similarity between users as
        weights on the rating given by the other user. Predictions can be based on the similarity between 
        known ratings or between latent features derived from a Singular Value Decomposition. 
           
        Paratmeters
        -----------
        curr_user : string
                    A unique identifier for a specific user, for whom the prediction will be made
                    
        song : string
                    A unique identifier for a specific song, for which the prediction will be made

        Returns
        -------
        overlap_prediction : int
                    The predicted rating, based on overlap similarity as ratings weights
        
        latent_prediction : int 
                    The predicted rating, based on latent similiarity as ratings weights 


        Notes
        -----
        Zeros are used for unsuccessful (not enough overlap) ratings because these will be unique (no 
        other combinations will create a 0 rating). This ensures that these can be removed from reporting.
        
        See Also 
        --------
        self.similar_users_to_user : class method to calculate similarity between users 
        """
        
        # randomise user if none is passed in 
        if (curr_user is None):
            curr_user = self.taste_profile['user'].values[random.randint(0, len(self.taste_profile['user']) - 1)]
        else:
            assert curr_user.isin(self.taste_profile['user'].values), "User not found, so unable to create recommendations"
            
        # randomise song if none is passed in 
        if (song is None):
            song = self.taste_profile['song'].values[random.randint(0, len(self.taste_profile['song']) - 1)]
        else:
            assert song.isin(self.taste_profile['song'].values), "User not found, so unable to create recommendations"
            
        # get similarities to current user (based on the 
        similarity_overlap, similarity_latent = self.similar_users_to_user(curr_user)
        overlap_predicted = 0 
        latent_predicted = 0 
        number_similar_considered = 0

        # loop over other users 
        for comparable_user in similarity_overlap.index.values:
            
            # get other user's rating
            other_rating = self.utility.loc[comparable_user, song]

            # if the other user rated 
            if np.isnan(other_rating) == False:
                
                # add to number of similarly users considered tally 
                number_similar_considered += 1 
                # add weighted rating to running total 
                overlap_predicted += similarity_overlap[comparable_user][0][0]*other_rating
                
                # if latent features are being considered add weighted latent rating to total 
                if len(similarity_latent) > 0:
                    latent_predicted += similarity_latent[comparable_user][0][0]*other_rating
        
        # if none are considered return 0 (this will be an indicator of unsuccessful ratings) 
        if number_similar_considered == 0:
            return 0, 0
        
        # Calculate the predictions by dividing total by number of predictions 
        else:
            overlap_prediction = overlap_predicted / number_similar_considered
            if latent_predicted == 0:
                latent_prediction = np.nan
                
            else:
                latent_prediction = latent_predicted / number_similar_considered
        
        # if latent predictions are not being considered only include the overlap predictions
        if (latent_prediction is None):
            return overlap_prediction
        
        else:
            return overlap_prediction, latent_prediction
                
           
    def generate_taste_profile_predictions(self, inclu_latent=True):
        """
        Creates predictions for all values within the taste profile (to find accuracy of overall 
        predictions) and returns a taste profile with all predictions. Can return both overlap
        based and latent based predictions. Count data is added from the original taste profile 
        for potential rank-based evaluations. Reports the number of combinations unable to predict
        but includes them in taste profile (as 0 rated).
        
        Returns
        -------
        full_taste_profile : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        Notes
        -----
        Zeros are used for unsuccessful (not enough overlap) ratings because these will be unique (no 
        other combinations will create a 0 rating). This ensures that these can be removed from reporting.
        
        See Also 
        --------
        self.user_rating_prediction : Creates predictions for each specific user-song combination passed in 
        
        """
        if (inclu_latent):
            assert len(self.user_matrix_df) > 0, "No user latent feature matrix passed into class intialisation" 

        # creates copy of taste profile to add to 
        tp_comparison = self.scaled_taste_profile.copy()
        
        # create matrix for user-song data
        rating_predictions = []
        
        # create array for only latent predictions
        latent_predictions = []

        # create an integer counter for songs unable to predict (no enough overlap users)
        unable = 0 
        
        # iterate over taste profile rows to create new predictions
        for triplet in tp_comparison.itertuples(): 
            user = triplet[1]
            song = triplet[2]
            scaled_score = triplet[3]
            count = self.taste_profile[(self.taste_profile['user'] == user) &
                    (self.taste_profile['song'] == song)]['count'].values[0]
            
            # get rating for specific user-song combination
            overlap, latent = self.user_rating_prediction(curr_user=user, song=song)
            
            # add to counter if unable to prediction
            if (overlap == 0):
                unable += 1
            
            # add current row to the matrix 
            rating_predictions.append([user, song, count, scaled_score, overlap])
            
            if (inclu_latent):
                latent_predictions.append(latent)
            
        # create dataframe of user-song combinations and count, rating and prediction
        full_taste_profile = pd.DataFrame(columns=['user', 'song', 'count', 'scaled_score', 'overlap_prediction'],
                                         data = rating_predictions)

        # add latent predictions if they are to be included
        if (inclu_latent):
            full_taste_profile['latent_predictions'] = latent_predictions
        
        # report the number of user-song combinations unable to predict
        print("Unable to predict ratings for:", unable, "user-song combinations")
        
        return full_taste_profile

    
    def recommend_user(self, rec_user=None, number_recs=9,
                              similarity_type='overlap'):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        
        # if using latent features, assert that they were passed into the class initialisation
        if (similarity_type=='latent'):
            assert len(self.user_matrix_df) > 0, "No user latent feature matrix passed into class intialisation" 

        # randomise user if none is passed in 
        if (rec_user is None):
            rec_user = self.taste_profile['user'].values[random.randint(0, len(self.taste_profile['user']) - 1)]
        else:
            assert rec_user.isin(self.taste_profile['user'].values), "User not found, so unable to create recommendations"
        
        # create user similarity series for the user to be recommended to 
        sim_series_overlap, sim_series_latent = self.similar_users_to_user(rec_user)
        
        # select series to use based on the similarity type 
        if (similarity_type == 'overlap'):
            sim_series = sim_series_overlap
        elif (similarity_type == 'latent') & (len(sim_series_latent) > 0):
            sim_series = sim_series_latent
        else:
            return "Incorrect similarity type or no latent user features passed in class initiation"
            
        # create array of similar users to consider and dataframe of songs     
        similar_enough = sim_series.index.values
        similar_user_songs = self.scaled_taste_profile[self.scaled_taste_profile['user'].isin(similar_enough)]

        song_reccs = pd.Series(dtype=float)
        
        for song in similar_user_songs['song'].unique():
            song_rated = similar_user_songs[similar_user_songs['song'] == song]
            
            song_score = 0
            
            # We don't want to be subject to the bias of one user?
            if (len(song_rated) > self.needed_similar_rating):
                for user in song_rated['user']:
                    song_score += sim_series[user]*song_rated[song_rated['user'] == user]['rating'].values[0]
            
                song_reccs[song] = song_score /len(song_rated)

        # get songs ids with the highest similiarity values 
        recs = song_reccs.sort_values(ascending=False).head(number_recs).index.values
        
        return self.song_info.loc[recs]
    
    
    
class SurpriseDecompositionRecommender:
    """
    This class acts as a Latent Features based recommendation system, by decomposing the taste profile for 
    user and song combinations into a latent feature user matrix and latent feature song matrix.
    
    On intialisation creates the framework necessary to perform a decomposition. This includes rating the listen
    counts (either all in one go or by user) from 1 to 5. The Surprise library, which is used to decompose the 
    data requires a Dataset object be passed into the SVD, and the Dataset object is built on user-item-rating
    pairs. These ratings must be within a fixed scale.
    
    Once the optimal decompositon is found, the best parameters can be fed to the funk_model function, allowing 
    the user to begin using the created latent user matrix (self.user_matrix_df) and latent song matrix 
    (self.song_matrix_df) to find song rating predictions (highest predicted songs will be reported as 
    recommendations for specific users.
    
    
    Parameters
    ----------
    taste_profile: Pandas DataFrame
                a Pandas DataFrame of a song taste profile - must include a count column - to be initialised
                as self.taste_profile
                
    
    song_info : Pandas DataFrame
                a Pandas DataFrame of the song title, artist name and release for recommendation output - to 
                be initialised as self.song_info
    
    rec_number : int
                The number of recommendations set to be given to a user - to be intialised into self.rec_number
    
    scale_by : string
                A string noting whether the scaling should be performed all together or by individual user, this
                is passed into the capstone_utils.scaling_mmb file 
                

    Notes
    -----
    The class must first be initialised before passing in the grid search parameters to find the optimal parameters
    for the SVD. (This is done separately so that if using a userbased scaling, the computational load to do so does 
    not need to be repeated).
    
    See Also
    --------
    capstone_utils.scaling_mmb : function for binning the listen counts into ratings (by user or all together)
    Surprise.SVD : SVD algorithm used to decompose the taste profile.
    """
    
    def __init__(self, taste_profile, song_info, rec_number=9, scale_by='allinone'):

        # Taste profile scaling
        self.taste_profile = taste_profile
        self.scaled_taste_profile = capstone_utils.scaling_mmb(self.taste_profile, scale_type=scale_by)        
        self.song_info = song_info
        self.rec_number = rec_number
        
        # readying data for a surprise utility matrix 
        self.Dataset_taste_profile = Dataset.load_from_df(self.scaled_taste_profile, reader = Reader(rating_scale=(1, 5)))
        
        # creating utility matrices for our training set and testing set
        self.utility = self.Dataset_taste_profile.build_full_trainset()
        
        
    def funk_decompose_search(self, param_grid, measures_=['rmse']):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        
        # from tempfile import mkdtemp
        # cachedir = mkdtemp() # stores the data for the cross validations in harddrive (slower) but not using RAM so heavily
        # pipe = Pipeline(estimators, memory = cachedir)
        

        # initialise grid search 
        grid_svd = GridSearchCVSurprise(FunkSVD, param_grid, measures=measures_, cv=5, n_jobs=-2)
        
        grid_svd.fit(self.Dataset_taste_profile)
        best_svd = grid_svd.best_estimator['rmse']
        print("Best RMSE:", grid_svd.best_score['rmse'])
        print("Optimal Parameters:\n",grid_svd.best_params['rmse'], sep='')
        
        self.best_params = grid_svd.best_params['rmse']


    def funk_model(self):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        funk = FunkSVD(n_factors = self.best_params['n_factors'], # number of 
                       n_epochs=self.best_params['n_epochs'], 
                       lr_all=self.best_params['lr_all'],    # Learning rate 
                       reg_all=self.best_params['reg_all'],
                       biased=False,  # forces all latent information to be stored 
                       verbose=0)
        
        # 
        self.funk_model = funk.fit(self.utility)
        
        self.user_matrix = funk.pu
        
        # Need to transpose so that we can compute the dot product in order to get the predicted ratings
        self.song_matrix = funk.qi.T 
        
        
        user_index = []
        for i in range(len(self.user_matrix)):
            user_index.append(self.utility.to_raw_uid(i))
            
        song_index = []
        for j in range(len(self.song_matrix.T)):
            song_index.append(self.utility.to_raw_iid(j))
    
        self.user_matrix_df = pd.DataFrame(data=self.user_matrix, index=user_index)
        self.song_matrix_df = pd.DataFrame(data=self.song_matrix, columns=song_index)
        
        # store predicted ratings - NOT WORKING
        self.predicted_ratings = self.user_matrix_df.dot(self.song_matrix_df)
        

    def generate_taste_profile_predictions(self):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        
        tp_comparison = self.scaled_taste_profile.copy()
        
        rating_predictions = []

        for triplet in tp_comparison.itertuples(): 
            user = triplet[1]
            song = triplet[2]
            scaled_score = triplet[3]
            count = self.taste_profile[(self.taste_profile['user'] == user) &
                    (self.taste_profile['song'] == song)]['count'].values[0]
            prediction = self.predicted_ratings[song][user]

            rating_predictions.append([user, song, count, scaled_score, prediction])

        full_taste_profile = pd.DataFrame(columns=['user', 'song', 'count', 'scaled_score', 'prediction'],
                                         data = rating_predictions)
        
        return full_taste_profile
        
    
    def single_song_latent_feature_graph(self, song_name=None):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        
        if (song_name is None):
            song_name = self.taste_profile['song'].values[random.randint(0, len(self.taste_profile['song']) - 1)]
        
        song_latent_feats = self.song_matrix_df.loc[:,song_name]
        
        plt.figure(figsize=(12, 4))
        plt.barh([f'{i}' for i in song_latent_feats.index], song_latent_feats)
        plt.title("Hidden Feature Composition of " + self.song_info.loc[song_name]['Title'], fontsize=20)
        plt.ylabel("Hidden Feature")
        plt.xlabel("Value")
        plt.yticks(ticks=None, labels=None)
        plt.show()

    
    def recommend_user(self, rec_user_id=None):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
            
        if (rec_user_id is None):
            rec_user_id = self.taste_profile['user'].values[random.randint(0, len(self.taste_profile['user']) - 1)]
            
        # encodings withing the trainset
        top_unrated_songs = self.predicted_ratings.loc[rec_user_id].drop(
            self.taste_profile[self.taste_profile['user'] == rec_user_id]
            ['song'].values).sort_values(ascending=False).head(self.rec_number).index
        
        return self.song_info.loc[top_unrated_songs]
        
        
    def recommend_all_users(self):
        """
        Recommends songs to a specific user based on a passed in similarity metric.
        
        Generates predicted song ratings from the songs that enough similar users have listened to. 
        Only songs that have been listened to by the similar users are considered for recommendation.
        Similarity type can be 'overlap' or 'latent'. 
        
        Parameters
        ----------
        rec_user : string
                    A unique identifier for the user to recommend songs to 
                    
        number_recs : int
                    The number of recommendations to return to the user

        
        Returns
        -------
        recs : Pandas DataFrame
                    A Pandas DataFrame with the user, song, rating and predicted rating(s) 

        See Also 
        --------
        self.similar_users_to_user : Finds similar enough users and reports the cosine similarity 
        
        """
        
        unique_users = self.taste_profile['user'].unique()
        user_recs = []
        for user in unique_users:
            user_recs.append(self.recommend_user(rec_user_id=user).index.values)
        
        user_recomendations = pd.DataFrame(columns=['user', 'recommended_songs'], 
                                           data = {'user': unique_users, 'recommended_songs': user_recs})

        return user_recomendations