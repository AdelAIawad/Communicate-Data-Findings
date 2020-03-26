#!/usr/bin/env python
# coding: utf-8

# In[3]:


# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.simplefilter("ignore")


# In[4]:


# load in the dataset into a pandas dataframe
loan = pd.read_csv('prosperLoanData.csv')

# Subset the dataframe by selecting features of interest
cols = ['LoanOriginalAmount', 'BorrowerAPR', 'StatedMonthlyIncome', 'Term', 'ProsperRating (Alpha)',  'EmploymentStatus']
loan_sub = loan[cols]

# remove loans with missing borrower APR information
loan_sub = loan_sub[~loan_sub.BorrowerAPR.isna()]
# remove loans with stated monthly income greater than 30k
loan_sub = loan_sub[loan_sub.StatedMonthlyIncome<=30000]
# Convert ProsperRating and Employment status into ordered categorical types
rate_order = ['HR','E','D','C','B','A','AA']
ordered_var = pd.api.types.CategoricalDtype(ordered = True,categories = rate_order)
loan_sub['ProsperRating (Alpha)'] = loan_sub['ProsperRating (Alpha)'].astype(ordered_var)

emp_order = ['Employed','Self-employed','Full-time','Part-time','Retired','Other','Not employed', 'Not available']
ordered_var = pd.api.types.CategoricalDtype(ordered = True,categories = emp_order)
loan_sub['EmploymentStatus'] = loan_sub['EmploymentStatus'].astype(ordered_var)
loan_sub.shape


# In[5]:


# The Distribution of Borrower APR multimodal. A little top focused at 0.1, an enormous pinnacle focused at 0.2. 
#There is additionally a little pinnacle focused 0.3. Furthermore, there is a very shape top somewhere in the range of 0.35 
#and 0.36. Truth be told, not many advances have APR more noteworthy than 0.43.


# In[6]:


plt.figure(figsize=[8, 5])
bins = np.arange(0, loan_sub.BorrowerAPR.max()+0.05, 0.01)
plt.hist(data = loan_sub, x = 'BorrowerAPR', bins = bins);
plt.xlabel('Borrower APR');
plt.title('Distribution of Borrower APR');


# In[7]:


# The Distribution of Original Loan Amount


# In[8]:


# The very large spikes in frequency are at 10k, 15k, 20k, 25k and 35k. There are also small spikes at
#8k,9k,11k,12k,13k,14k etc. It means that most of the loans are multiples of 1k.


# In[9]:


plt.figure(figsize=[8, 5])
bins = np.arange(8000, loan_sub.LoanOriginalAmount.max()+300, 300)
plt.hist(data = loan_sub, x = 'LoanOriginalAmount', bins = bins);
plt.xlabel('Original loan amount ($)');
plt.title('Distribution of Original Loan Amount');


# In[10]:


# Loan Amount vs Borrower APR 


# In[13]:


plt.figure(figsize = [8, 6])
sb.regplot(data = loan_sub, x = 'LoanOriginalAmount', y = 'BorrowerAPR', scatter_kws={'alpha':0.01});
plt.xlabel('Loan Amount ($)')
plt.ylabel('Borrower APR')
plt.title('Borrower APR vs. Loan Amount');


# In[12]:


# Borrower APR vs. Prosper Rating


# In[14]:


#The borrower APR decreases with the inexorably better evaluating. Borrowers with the best Prosper 
# appraisals have the most minimal APR. It implies that the Prosper rating strongly affects borrower APR.


# In[16]:


plt.figure(figsize=[8,6])
default_color = sb.color_palette()[0]
sb.boxplot(data=loan_sub, x='ProsperRating (Alpha)', y='BorrowerAPR', color=default_color)
plt.xlabel('Prosper Rating')
plt.ylabel('Borrower APR')
plt.title(' Prosper Rating vs Borrower APR ');


# In[17]:


#Prosper Rating Effect on Relationship between APR and Loan Amount#


# In[18]:


# The advance sum increments with better evaluating. The borrower APR diminishes with better evaluating. Strikingly,
#the connection between borrower APR and advance sum abandons negative to marginally positive when the Prosper appraisals
#are expanded from HR to An or better. This is may in light of the fact that individuals with An or AA appraisals will in
#general get more cash, expanding APR could counteract them acquire significantly more and amplify the benefit. Be that as
#it may, individuals with lower appraisals will in general obtain less cash, diminishing APR could urge them to acquire more.


# In[23]:


g=sb.FacetGrid(data=loan_sub,col='ProsperRating (Alpha)', height=3.5, col_wrap=4)
g.map(sb.regplot, 'LoanOriginalAmount', 'BorrowerAPR', x_jitter=0.04, scatter_kws={'alpha':0.1});
g.set_titles('{col_name}')
g.add_legend();
g.set_xlabels('Loan Amount ($)')
g.set_ylabels('Borrower APR')
plt.subplots_adjust(top=0.85)
plt.suptitle('Prosper Rating Effect on Relationship between APR and Loan Amount');


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




