import numpy as np

def create_features(data, col):
    data['dayofweek'] = data[col].dt.dayofweek
    data['month'] = data[col].dt.month
    data['year'] = data[col].dt.year
    data['quarter'] = data[col].dt.quarter
    data['dayofyear'] = data[col].dt.dayofyear
    data['weekofyear'] = data[col].dt.isocalendar().week
    data['dayofmonth'] = data[col].dt.day
    data['hour'] = data[col].dt.hour
    data['minute'] = data[col].dt.minute
    data['second'] = data[col].dt.second
    data['horizon'] = (data[col] - data[col].min()) / np.timedelta64(1, 's')

    X = data[['year','quarter','month','dayofweek',
           'dayofyear','dayofmonth','weekofyear',
            'hour', 'minute', 'second', 'horizon']]

    X.columns = [str(x) + "_" + col for x in X.columns]
    
    return X