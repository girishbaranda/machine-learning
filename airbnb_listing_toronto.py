import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

calendar  = pd.read_csv('calendar.csv.gz')
print('We have', calendar.date.nunique(), 'days and', calendar.listing_id.nunique(), 'unique listings in the calendar data.')

print(calendar.date.min(), calendar.date.max())

# print(calendar.isnull.sum())

# print(calendar.shape())

# print(calendar.head())

# print(calendar.available.value_counts())

calendar_new = calendar[['date', 'available']]
calendar_new['busy'] = calendar_new.available.map( lambda x: 0 if x == 't' else 1)
calendar_new = calendar_new.groupby('date')['busy'].mean().reset_index()
calendar_new['date'] = pd.to_datetime(calendar_new['date'])

plt.figure(figsize=(10, 5))
plt.plot(calendar_new['date'], calendar_new['busy'])
plt.title('Airbnb Amsterdam Calendar')
plt.ylabel('% busy')
plt.show()

