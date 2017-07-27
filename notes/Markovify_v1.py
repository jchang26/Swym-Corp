import pandas as pd
from itertools import product
import numpy as np

class Markovify(object):
    def __init__(self, order = 1, current_state =  1):
        self.order = order
        self.current_state = current_state
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
        self.possible_events = [-1, 1, 3, 4, 6, 7, 8, 104]
        '''
        self.events_row_map = {
            -1 : 8
            ,0 : 0
            ,1 : 1
            ,3 : 2
            ,4 : 3
            ,6 : 6
            ,7 : 7
            ,8 : 4
            ,100 : 9
            ,104 : 5
        }
        '''
        self.events_row_map = {
            -1 : 7
            ,1 : 0
            ,3 : 1
            ,4 : 2
            ,6 : 5
            ,7 : 6
            ,8 : 3
            ,104 : 4
        }
        self.events_desc = {
            -1: 'Delete from Wishlist'
            ,1: 'Page View'
            ,3: 'Add to Cart'
            ,4: 'Add to Wishlist'
            ,6: 'Purchase'
            ,7: 'Remove from Cart'
            ,8: 'Add to Watchlist'
            ,104: 'Begin Checkout'
            ,0: 'Session Start'
            ,100: 'Session End'
        }
        self.row_legend = {}
        self.Markov_mat = None
        self.action_counts = []
        self.session_times = []

    def fit(self, filename):
        #Read in Sessions .csv file
        session = pd.read_csv(filename, header = None)
        session.columns = self.session_columns

        #Drop unused variables
        session.drop(['category','imageurl','pagetitle','pageurl','userid'
                     ,'fullurl','providerid','productid','normalizedpageurl'
                     ,'rawpageurl','referrerurl','rawreferrerurl','utmsource'
                     ,'utmmedium','utmcontent','utmcampaign','utmterm'
                     ,'ipaddress','deviceid','requesttype','quantity','price']
                     ,axis = 1, inplace = True)

        #Format date/time variables
        session['createddate'] = pd.to_datetime(session['createddate'])

        #Drop sessionID NA's, since can't be sure they're always same session
        session = session[session['sessionid'].notnull()]

        #Initialize dictionary for possible transition counts
        events_dict = {}
        for i in product(self.possible_events,repeat=self.order+1):
            events_dict[i] = 0

        #Add to transition counts
        for j in session['sessionid'].unique():
            one_session = session[session['sessionid'] == j].sort_values('createddate')
            self.action_counts.append(one_session.shape[0])
            #begin = (0,one_session['eventtype'].iloc[0])
            #end = (one_session['eventtype'].iloc[one_session.shape[0]-1],100)
            #events_dict[begin] += 1
            #events_dict[end] += 1
            for k in range(one_session.shape[0]-2):
                events_dict[(one_session['eventtype'].iloc[k],one_session['eventtype'].iloc[k+1])] += 1

        #convert to numpy matrix of transition counts
        np_events_dict = {}
        for k,v in events_dict.items():
            mapped_key = (self.events_row_map[k[0]],self.events_row_map[k[1]])
            np_events_dict[mapped_key] = v

        transition_counts = np.zeros((len(self.possible_events),len(self.possible_events)))
        for x,y in np_events_dict.items():
            transition_counts[x[0]][x[1]] = y

        #Convert to transition probability matrix
        #transition_counts[self.events_row_map[100]][self.events_row_map[100]] = 1
        self.Markov_mat = transition_counts.astype(float) / transition_counts.sum(axis = 1,keepdims = True)

        #Initialize legend for matrix
        for a,b in self.events_row_map.items():
            self.row_legend[b] = self.events_desc[a]

    def predict_next(self):
        current_mat = np.zeros(len(self.possible_events))
        current_mat[self.events_row_map[self.current_state]] = 1
        result = (current_mat * self.Markov_mat.T).T
        max_idx = result.argmax()
        idx_tup = np.unravel_index(max_idx, result.shape)
        self.current_state = [key for key, value in self.events_row_map.items() if value == idx_tup[1]][0]
        return self.events_desc[self.current_state]

    def score(self, filename):
        pass

if __name__ == '__main__':
    test = Markovify(current_state = 3)
    test.fit('data/session_data_sample_030113.csv')
