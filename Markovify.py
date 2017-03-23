import pandas as pd
import numpy as np
from urlparse import urlparse

class Markovify(object):
    def __init__(self, order = 1):
        self.order = order
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
        self.possible_events = [-1, 1, 3, 4, 6, 7, 8, 104]

        def clean_swym_device_data(device):

            df = device.copy()

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

            return df


if __name__ == '__main__':
    session = pd.read_csv('data/session_data_sample_030113.csv', header = None)
    device = pd.read_csv('data/devices_data_sample_030113.csv', header = None)
