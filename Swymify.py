import pandas as pd
import numpy as np
from urlparse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

class Swymify(object):
    def __init__(self, order = 1, subset = 1.0):
        #Order of prior actions to be derived
        self.order = order

        #Proportion of dataset to consider, for speed of calculation purposes
        self.subset = subset

        #Standard column names for session and device data
        self.session_columns  = [
            'sessionid',
            'category',
            'imageurl',
            'createddate',
            'pagetitle',
            'pageurl',
            'userid',
            'fullurl',
            'providerid',
            'productid',
            'normalizedpageurl',
            'rawpageurl',
            'referrerurl',
            'rawreferrerurl',
            'utmsource',
            'utmmedium',
            'utmcontent',
            'utmcampaign',
            'utmterm',
            'ipaddress',
            'deviceid',
            'requesttype',
            'eventtype',
            'quantity',
            'price'
        ]
        self.device_columns = [
            'deviceid',
            'devicecategory',
            'devicetype',
            'agenttype',
            'os',
            'osversion',
            'useragent',
            'providerid',
            'createddate',
            'userid',
            'authtype'
        ]

        #Possible events dictionary
        self.events_desc = {
            -1: 'Delete from Wishlist',
            1: 'Page View',
            3: 'Add to Cart',
            4: 'Add to Wishlist',
            6: 'Purchase',
            7: 'Remove from Cart',
            8: 'Add to Watchlist',
            104: 'Begin Checkout'
        }

        #Initialize final output modeling matrices
        self.swym_x = None
        self.swym_y = None

        #Initialize tf-idf vectors
        self.referrer_tfidf = None
        self.category_tfidf = None
        self.page_tfidf = None

        #Testing dataset matrices
        self.test_x = None
        self.test_y = None

        #Model final selection
        self.model = None

        #First order Markov Chain transition matrix and row/column key
        self.Markov_mat = None
        self.Markov_classes = None

        #Intermediate step for testing and validation of code, and single session initialization
        self.modeling_df = None

    def swym_subset_data(self, data):
        #Split data if applicable
        df = data.copy()
        unique_sessions = df['sessionid'].unique()
        train_sess, test_sess = train_test_split(unique_sessions, test_size = self.subset)
        train = df[df['sessionid'].isin(train_sess)]
        test = df[df['sessionid'].isin(test_sess)]

        return test

    def swym_prelim_clean(self, session):
        #Preliminary scrubbing of session data, no complicated feature engineering
        df = session.copy()

        #Drop unnecessary columns
        df.drop(['imageurl','pageurl','fullurl','normalizedpageurl','rawpageurl','rawreferrerurl'
                ,'utmsource','utmmedium','utmcontent','utmcampaign','utmterm','ipaddress','requesttype']
                ,axis = 1, inplace = True)

        #Drop null sessionid, createddate and eventtype
        #Affect ability to derive predicted variable
        df = df[df['sessionid'].notnull()]
        df = df[df['createddate'].notnull()]
        df = df[df['eventtype'].notnull()]

        #Preliminary feature formatting and engineering
        df['category'] = df['category'].fillna('')
        df['createddate'] = pd.to_datetime(df['createddate'])
        df['dayofweek'] = df['createddate'].dt.dayofweek
        df['hour'] = df['createddate'].dt.hour
        df['pagetitle'] = df['pagetitle'].fillna('')
        df['providerid'] = df['providerid'].fillna('Unknown')
        df['productid'] = df['productid'].fillna(0.0)
        df['referrerurl'] = df['referrerurl'].fillna('')
        df['referrerurl'] = df['referrerurl'].apply(urlparse)
        df['referrerurl'] = df['referrerurl'].apply(lambda x: x.netloc)
        df['deviceid'] = df['deviceid'].fillna('Unknown')
        df['quantity'] = df['quantity'].fillna(0.0)
        df['price'] = df['price'].fillna(0.0)

        return df

    def swym_clean_device(self, session, device):
        #Prep device data and join onto session data by deviceid
        df = device.copy()

        #Group "Other" entries in relevant features, drop unnecessary columns
        category_list = ['iPhone','Windows PC','Android phone','Mac','iPad','Linux PC'
                        ,'Android PC','Android tablet','Windows phone']
        df['devicecategory'] = df['devicecategory'].apply(lambda x: x if x in category_list else 'Other')

        type_list = ['Smartphone','Personal computer', 'Tablet']
        df['devicetype'] = df['devicetype'].apply(lambda x: x if x in type_list else 'Other')

        agent_list = ['Mobile Browser','Browser']
        df['agenttype'] = df['agenttype'].apply(lambda x: x if x in agent_list else 'Other')

        os_list = ['iOS','Android','Windows','OS X', 'Linux']
        df['os'] = df['os'].apply(lambda x: x if x in os_list else 'Other')

        df.drop(['osversion','useragent','providerid','createddate','userid','authtype']
               , axis = 1, inplace = True)
        df = df[df.notnull()]

        #Can have multiple users per device, but since only want device info can ignore
        df.drop_duplicates(subset = 'deviceid', keep = 'first', inplace = True)

        #Join device data onto session data
        df.set_index('deviceid', inplace = True)
        session = session.join(df, on = 'deviceid', how = 'left')
        session['devicecategory'] = session['devicecategory'].fillna('Unknown')
        session['devicetype'] = session['devicetype'].fillna('Unknown')
        session['agenttype'] = session['agenttype'].fillna('Unknown')
        session['os'] = session['os'].fillna('Unknown')

        return session

    def swym_next_action(self, data):
        #Creates predicted next action and session history variables
        df = data.copy()

        #Initialize empty dataframe to attach each session onto
        output_columns = list(df.columns)
        output_columns.append('elapsedtime')
        output_columns.append('totalelapsedtime')
        for o in range(self.order-1):
            col_name = str(o+1)+'prioraction'
            output_columns.append(col_name)
        output_columns.append('nextaction')
        output = pd.DataFrame(columns = output_columns)

        #For each session, calculate time since last action, total elapsed session time
        #Prior actions depending on order of class
        for i in df['sessionid'].unique():
            one_session = df[df['sessionid'] == i].sort_values('createddate')
            elapsedtime = np.zeros(one_session.shape[0],dtype = int)
            totalelapsedtime = np.zeros(one_session.shape[0],dtype = int)
            prioraction_dict = {}
            for o in range(self.order-1):
                prioraction_name = str(o+1)+'prioraction'
                prioraction_dict[prioraction_name] = np.zeros(one_session.shape[0],dtype = int)
            nextaction = np.zeros(one_session.shape[0],dtype = int)
            for j in range(one_session.shape[0]):
                if j > 0:
                    timedelta = one_session['createddate'].iloc[j]-one_session['createddate'].iloc[j-1]
                    totaltimedelta = one_session['createddate'].iloc[j]-one_session['createddate'].iloc[0]
                    elapsedtime[j] = (timedelta/np.timedelta64(1,'s')).astype(int)
                    totalelapsedtime[j] = (totaltimedelta/np.timedelta64(1,'s')).astype(int)
                for o in range(self.order-1):
                    prioraction_name = str(o+1)+'prioraction'
                    if j > o:
                        prioraction_dict[prioraction_name][j] = one_session['eventtype'].iloc[j-o-1]
                if j < one_session.shape[0]-1:
                    nextaction[j] = one_session['eventtype'].iloc[j+1]

            #Append new columns to each session
            one_session['elapsedtime'] = elapsedtime
            one_session['totalelapsedtime'] = totalelapsedtime
            for o in range(self.order-1):
                col_name = str(o+1)+'prioraction'
                one_session[col_name] = prioraction_dict[col_name]
            one_session['nextaction'] = nextaction

            #Drop rows missing necessary modeling information and append to output
            for o in range(self.order-1):
                col_name = str(o+1)+'prioraction'
                one_session = one_session[one_session[col_name] != 0]
            one_session = one_session[one_session['nextaction'] != 0]
            output = output.append(one_session, ignore_index = True)

        return output

    def swym_dummy_featurize(self, data):
        #Create dummy features for data
        df = data.copy()

        #Create dummy variables for session data
        for i, j in self.events_desc.items():
            df[j] = df['eventtype'].apply(lambda x: 1 if x == i else 0)
            for o in range(self.order-1):
                event_name = j + ' ' + str(o+1)
                prioraction_name = str(o+1)+'prioraction'
                df[event_name] = df[prioraction_name].apply(lambda x: 1 if x == i else 0)
        for q in range(self.order-1):
            event_name = 'Add to Watchlist '+str(q+1)
            df.drop(event_name, axis = 1, inplace = True)
        df.drop('Add to Watchlist', axis = 1, inplace = True)

        dow_desc = {
            0.0: 'Monday',
            1.0: 'Tuesday',
            2.0: 'Wednesday',
            3.0: 'Thursday',
            4.0: 'Friday',
            5.0: 'Saturday',
            6.0: 'Sunday'
        }
        for i, j in dow_desc.items():
            df[j] = df['dayofweek'].apply(lambda x: 1 if x == i else 0)
        df.drop('Monday', axis = 1, inplace = True)

        hour_desc = {}
        for a in range(24):
            hour_desc[float(a)] = 'Hour '+str(a)
        for i, j in hour_desc.items():
            df[j] = df['hour'].apply(lambda x: 1 if x == i else 0)
        df.drop('Hour 0', axis = 1, inplace = True)

        #Device dummies
        category_list = ['iPhone','Windows PC','Android phone','Mac','iPad','Linux PC'
                        ,'Android PC','Android tablet','Windows phone']
        for a in category_list:
            df[a] = df['devicecategory'].apply(lambda x: 1 if x == a else 0)

        type_list = ['Smartphone','Personal computer', 'Tablet']
        for a in type_list:
            df[a] = df['devicetype'].apply(lambda x: 1 if x == a else 0)

        agent_list = ['Mobile Browser','Browser']
        for a in agent_list:
            df[a] = df['agenttype'].apply(lambda x: 1 if x == a else 0)

        os_list = ['iOS','Android','Windows','OS X', 'Linux']
        for a in os_list:
            df[a] = df['os'].apply(lambda x: 1 if x == a else 0)

        return df

    def swym_nlp_featurize(self, data):
        #Create nlp features, important word counts
        df = data.copy()

        #Fit tf-idf vectors for applicable columns
        self.referrer_tfidf = TfidfVectorizer(stop_words = 'english', max_features = 100)
        self.referrer_tfidf.fit(df['referrerurl'])
        referrer_vect = self.referrer_tfidf.transform(df['referrerurl'])
        referrer_columns = self.referrer_tfidf.get_feature_names()
        referrer_df = pd.DataFrame(referrer_vect.toarray(), columns = referrer_columns)
        df = pd.concat([df,referrer_df], axis = 1)

        self.category_tfidf = TfidfVectorizer(stop_words = 'english', max_features = 100)
        self.category_tfidf.fit(df['category'])
        category_vect = self.category_tfidf.transform(df['category'])
        category_columns = self.category_tfidf.get_feature_names()
        category_df = pd.DataFrame(category_vect.toarray(), columns = category_columns)
        df = pd.concat([df,category_df], axis = 1)

        self.page_tfidf = TfidfVectorizer(stop_words = 'english', max_features = 100)
        self.page_tfidf.fit(df['pagetitle'])
        page_vect = self.page_tfidf.transform(df['pagetitle'])
        page_columns = self.page_tfidf.get_feature_names()
        page_df = pd.DataFrame(page_vect.toarray(), columns = page_columns)
        df = pd.concat([df,page_df], axis = 1)

        return df

    def swym_nlp_read(self, data):
        #Create nlp features, important word counts for test data
        df = data.copy()

        #Attach same tf-idf vectors for applicable columns to test data
        referrer_vect = self.referrer_tfidf.transform(df['referrerurl'])
        referrer_columns = self.referrer_tfidf.get_feature_names()
        referrer_df = pd.DataFrame(referrer_vect.toarray(), columns = referrer_columns)
        df = pd.concat([df,referrer_df], axis = 1)

        category_vect = self.category_tfidf.transform(df['category'])
        category_columns = self.category_tfidf.get_feature_names()
        category_df = pd.DataFrame(category_vect.toarray(), columns = category_columns)
        df = pd.concat([df,category_df], axis = 1)

        page_vect = self.page_tfidf.transform(df['pagetitle'])
        page_columns = self.page_tfidf.get_feature_names()
        page_df = pd.DataFrame(page_vect.toarray(), columns = page_columns)
        df = pd.concat([df,page_df], axis = 1)

        return df

    def swym_trim_data(self, data):
        #Delete extraneous columns that were otherwise featurized
        df = data.copy()

        #Dependent Variable
        y = df['nextaction']

        #Drop variables
        df.drop(['sessionid','createddate','userid','deviceid','nextaction','providerid','productid'
                ,'referrerurl','category','pagetitle'
                ,'eventtype','dayofweek','hour'
                ,'devicecategory','devicetype','agenttype','os']
                , axis = 1, inplace = True)
        for q in range(self.order-1):
            prioraction_name = str(q+1)+'prioraction'
            df.drop(prioraction_name, axis = 1, inplace = True)

        return df, y

    def swym_prior_history(self, data):
        #Check time frame for prior sessions
        df = data.copy()

        #Unique user sessions, ordered by time session began
        df['identifier'] = df['sessionid'] + df['userid']
        trunc = df[['sessionid','userid','createddate']]
        trunc = trunc.groupby(['sessionid','userid'], as_index = False).agg({'createddate': 'min'})
        trunc = trunc.sort_values(['userid','createddate'])

        #If prior session exists in time frame for user, then has history
        prior_hist = np.zeros(trunc.shape[0], dtype = int)
        for row in range(trunc.shape[0]):
            if row > 0:
                if trunc['userid'].iloc[row] == trunc['userid'].iloc[row-1]:
                    prior_hist[row] = 1
        trunc['hist_ind'] = prior_hist

        #Join back on by user session ID
        trunc['identifier'] = trunc['sessionid'] + trunc['userid']
        trunc.drop(['sessionid','userid','createddate'],axis = 1, inplace = True)
        trunc.set_index('identifier', inplace = True)
        df = df.join(trunc, on = 'identifier', how = 'left')
        df.drop('identifier',axis = 1, inplace = True)
        df['hist_ind'] = df['hist_ind'].fillna(0)
        df['userid'] = df['userid'].fillna('Unknown')

        return df

    def swym_load_data(self, session_path, device_path):
        #Read in swym session and device data for some time period for training
        session = pd.read_csv(session_path, header = None)
        session.columns = self.session_columns
        device = pd.read_csv(device_path, header = None)
        device.columns = self.device_columns
        df = session.copy()
        df2 = device.copy()

        #Preliminary cleaning and feature engineering
        df = self.swym_prelim_clean(df)
        #Join on device data
        df = self.swym_clean_device(df, df2)
        #Add on prior session history indicator
        df = self.swym_prior_history(df)
        #Subset for speed, if applicable
        if self.subset != 1.0:
            df = self.swym_subset_data(df)
        #Add on next action predicted variable and prior actions if applicable
        self.modeling_df = self.swym_next_action(df)
        #Replace categorical variables with dummies
        df = self.swym_dummy_featurize(self.modeling_df)
        #Fit tf-idf columns
        df = self.swym_nlp_featurize(df)
        #Output final modeling data
        self.swym_x, self.swym_y = self.swym_trim_data(df)

    def swym_read_new(self, session_path, device_path):
        #Read in swym session and device data for some time period for testing
        session = pd.read_csv(session_path, header = None)
        session.columns = self.session_columns
        device = pd.read_csv(device_path, header = None)
        device.columns = self.device_columns
        df = session.copy()
        df2 = device.copy()

        #Preliminary cleaning and feature engineering
        df = self.swym_prelim_clean(df)
        #Join on device data
        df = self.swym_clean_device(df, df2)
        #Add on prior session history indicator
        df = self.swym_prior_history(df)
        #Add on next action predicted variable and prior actions if applicable
        df = self.swym_next_action(df)
        #Replace categorical variables with dummies
        df = self.swym_dummy_featurize(df)
        #Apply fitted tf-idf columns
        df = self.swym_nlp_read(df)
        #Output final test data
        self.test_x, self.test_y = self.swym_trim_data(df)

    def rfc_test(self):
        #Evaluate Random Forest via 5-folds cross validated score
        rfc = RandomForestClassifier(max_features = None, max_depth = 5, n_jobs = -1)
        return np.mean(cross_val_score(rfc,self.swym_x,self.swym_y,cv=5))

    def gbc_test(self):
        #Evaluate Gradient Boosting via 5-folds cross vexamplalidated score
        gbc = GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 1000)
        return np.mean(cross_val_score(gbc,self.swym_x,self.swym_y,cv=5))

    def rfc_fit(self):
        #Fit random forest classifier from overall training data with gridsearched hyperparameters
        self.model = RandomForestClassifier(max_features = None, max_depth = 5, n_jobs = -1)
        self.model.fit(self.swym_x, self.swym_y)

    def rfc_score(self):
        #Score accuracy of selected random forest model on test dataset
        return self.model.score(self.test_x, self.test_y)

    def markovify(self, single_session):
        #Pass in initial session and corresponding device data
        #Build 1st order markov transition matrix
        if self.order != 1:
            print 'Must be first order Markov Chain'
        else:
            #Start from preloaded data
            df = self.modeling_df.copy()
            #Only use single session data
            df = df[df['sessionid'] == single_session].sort_values('createddate').head(1)
            df.reset_index(drop = True, inplace = True)
            #Replace categorical variables with dummies
            df = self.swym_dummy_featurize(df)
            #Apply fitted tf-idf columns
            df = self.swym_nlp_read(df)
            #Output single session formatted data as dictionary
            single_x, single_y = self.swym_trim_data(df)
            Markov_dict = {}
            for i in self.events_desc.values():
                Markov_x = single_x.copy()
                if i != 'Add to Watchlist':
                    Markov_x[i] = 1
                    for j in self.events_desc.values():
                        if j != 'Add to Watchlist' and j != i:
                            Markov_x[j] = 0
                Markov_dict[i] = self.model.predict_proba(Markov_x)
            #Change to numpy matrix for easier use
            self.Markov_classes = np.empty(len(self.model.classes_), dtype = str)
            for j in range(len(self.Markov_classes)):
                self.Markov_classes[j] = self.events_desc[self.model.classes_[j]]
            listify = []
            for k in self.Markov_classes:
                listify.append[Markov_dict[k]]
            self.Markov_mat = np.matrix(listify)
            return self.Markov_mat, Markov_classes

if __name__ == '__main__':
    '''
    for i in range(6):
        example = Swymify(order = i+1)
        example.swym_load_data('data/session_data_training_feb.csv', 'data/devices_data_training_feb.csv')
        print 'Order ' + str(i+1) + ' RFC cross-validation score = ' + str(example.rfc_test())
        example.rfc_fit()
        example.swym_read_new('data/session_data_test_march.csv','data/devices_data_test_march.csv')
        print 'Order ' + str(i+1) + ' RFC testing score = ' + str(example.rfc_score())
    '''
    example = Swymify()
    example.swym_load_data('data/session_data_training_feb.csv', 'data/devices_data_training_feb.csv')
    print example.rfc_test()
    example.rfc_fit()
    example.swym_read_new('data/session_data_test_march.csv','data/devices_data_test_march.csv')
    print example.rfc_score()
    example_mat, example_class = example.markovify('8k5novj0knl8vha42324qs62p6vfa3im8djvdctm3m8p6gqkdwpnu632xpczvo5r')
