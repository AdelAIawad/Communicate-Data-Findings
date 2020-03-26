#!/usr/bin/env python
# coding: utf-8

# In[2]:



# import all packages and set plots to be embedded inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:



# load in the dataset into a pandas dataframe, print statistics
loan = pd.read_csv('prosperLoanData.csv')


# In[4]:



# high-level overview of data shape and composition
print(loan.shape)
print(loan.head(6))


# In[5]:



# Subset the dataframe by selecting features of interest
cols = ['LoanOriginalAmount', 'BorrowerAPR', 'StatedMonthlyIncome', 'Term', 'ProsperRating (Alpha)', 
        'EmploymentStatus']
loan_sub = loan[cols]


# In[6]:



loan_sub.head()


# In[7]:



# descriptive statistics for numeric variables
loan_sub.describe()


# In[8]:


# Remove loans with missing borrower APR information
loan_sub = loan_sub[~loan_sub.BorrowerAPR.isna()]


# In[9]:


#What is the structure of your dataset?


# In[10]:


#The dataset contains 113,937 credits with 81 highlights (counting LoanOriginalAmount,
#BorrowerAPR, StatedMonthlyIncome, Term, ProsperRating (Alpha), EmploymentStatus and numerous others)


# In[11]:


#What is/are the main feature(s) of interest in your dataset?


# In[12]:


#I'm most interested in figureing out what features are best for predicting the borrower's Annual Percentage Rate (APR) for the loan


# In[13]:


get_ipython().set_next_input('What features in the dataset do you think will help support your investigation into your feature(s) of interest');get_ipython().run_line_magic('pinfo', 'interest')


# In[ ]:


What features in the dataset do you think will help support your investigation into your feature(s) of interest


# In[14]:


#I expect that the absolute advance sum will negatively affect the APR of the advance: the bigger the 
#complete credit sum, the lower the APR. I likewise imagine that the borrowers expressed month to month salary0,
#advance term, Prosper rating, business status will likewise have consequences for the APR


# In[15]:



I'll start by looking at the distribution of the main variable of interest: borrower APR.


# In[ ]:


bins = np.arange(0, loan_sub.BorrowerAPR.max()+0.05, 0.01)
plt.figure(figsize=[8, 5])
plt.hist(data = loan_sub, x = 'BorrowerAPR', bins = bins);
plt.xlabel('Borrower APR');


# In[ ]:


The appropriation of APR looks multi-modular. A little top focused at 0.1, an enormous pinnacle focused at 0.2.
There is likewise a little pinnacle focused 0.3. Also, there is a very shape top somewhere in the range of 0.35 and 0.36. 
Without a doubt, not many credits have APR more prominent than 0.43


# In[ ]:



# Check loans with APR greater than 0.43
loan_sub[loan_sub.BorrowerAPR>0.43]


# In[ ]:


#The 6 borrowers with biggest APR have little credit sum and don't have records of Prosper rating and work status. 

#Next up, take a gander at the conveyance of the main indicator variable of intrigue: LoanOriginalAmount


# In[ ]:



bins = np.arange(8000, loan_sub.LoanOriginalAmount.max()+200, 200)
plt.figure(figsize=[8, 5])
plt.hist(data = loan_sub, x = 'LoanOriginalAmount', bins = bins);
plt.xlabel('Original loan amount ($)');


# In[ ]:


#The extremely enormous spikes in recurrence are at 10k, 15k, 20k, 25k and 35k. There are additionally little spikes at 8k,9k,11k,12k,13k,14k and so forth. It implies that a large portion of the advances are products of 1k. 
#Presently, take a gander at the conveyances of different factors of intrigue: expressed month to month pay


# In[ ]:


# Distribution of stated monthly income
bins_smi = np.arange(0, 50000, 500)
plt.hist(data = loan_sub, x = 'StatedMonthlyIncome', bins=bins_smi);


# In[ ]:


# Check borrowers with stated monthly income greater than 1e5
loan_sub[loan_sub.StatedMonthlyIncome>1e5]


# In[ ]:


# Get percent of borrowers whose stated monthly income greater than 30k
(loan_sub.StatedMonthlyIncome>30000).sum()/float(loan_sub.shape[0])


# In[ ]:


# remove loans with stated monthly income greater than 30k, which are outliers
loan_sub = loan_sub[loan_sub.StatedMonthlyIncome<=30000]


# In[ ]:


(loan_sub.StatedMonthlyIncome>30000).sum()


# In[ ]:


#Look at distributions of term, Prosper rating and employment status


# In[ ]:


rate_order = ['HR','E','D','C','B','A','AA']
ordered_var = pd.api.types.CategoricalDtype(ordered = True,
                                    categories = rate_order)
loan_sub['ProsperRating (Alpha)'] = loan_sub['ProsperRating (Alpha)'].astype(ordered_var)

emp_order = ['Employed','Self-employed','Full-time','Part-time','Retired','Other','Not employed', 'Not available']
ordered_var = pd.api.types.CategoricalDtype(ordered = True,
                                    categories = emp_order)
loan_sub['EmploymentStatus'] = loan_sub['EmploymentStatus'].astype(ordered_var)
# Convert ProsperRating and Employment status into ordered categorical types


# In[ ]:



fig, ax = plt.subplots(nrows=3, figsize = [8,8])
default_color = sb.color_palette()[0]
sb.countplot(data = loan_sub, x = 'Term', color = default_color, ax = ax[0])
sb.countplot(data = loan_sub, x = 'ProsperRating (Alpha)', color = default_color, ax = ax[1])
sb.countplot(data = loan_sub, x = 'EmploymentStatus', color = default_color, ax = ax[2]);
plt.xticks(rotation=30);


# In[ ]:


#The length of a large portion of the advances are three years. The evaluations of the greater part of the borrowers
#are among D to A. The greater part of borrowers are utilized and full-time


# In[ ]:


#Of the highlights you explored, were there any bizarre circulations? Did you play out any activities on the information
#to clean, alter, or change the type of the information? Assuming this is the case, for what reason did you do this?


# In[ ]:


#the disseminations of expressed month to month pay is exceptionally right screwed. Most expressed month to month earnings are
#under 30k, however some of them are fantastically high, as more noteworthy than 100k. Shockingly, the greater part of borrowers
#with more noteworthy than 100k month to month pay just advance under 5k dollars. In this way, the enormous expressed month to
#month salary might be made up. In general, Less than 0.3 percent borrowers have expressed month to month salary more noteworthy
#than 30k, these can be appeared as exception for the accompanying investigation, so it is smarter to evacuate borrower records
#with pay more prominent than 30k.

#There is no need to perform any transformations


# In[16]:



num_vars = ['LoanOriginalAmount', 'BorrowerAPR', 'StatedMonthlyIncome']
cat_vars = ['Term', 'ProsperRating (Alpha)', 'EmploymentStatus']


# In[17]:



# correlation plot
plt.figure(figsize = [8, 5])
sb.heatmap(loan_sub[num_vars].corr(), annot = True, fmt = '.3f',
           cmap = 'vlag_r', center = 0);


# In[18]:



# plot matrix: sample 5000 loans so that plots are clearer and render faster
loan_sub_samp = loan_sub.sample(5000)
g = sb.PairGrid(data = loan_sub_samp.dropna(), vars = num_vars)
g = g.map_diag(plt.hist, bins=20)
g.map_offdiag(plt.scatter, alpha=0.2);


# In[19]:


#the relationship coefficient of borrower APR and advance unique sum is - 0.323, the disperse plot likewise shows that these 
#two factors are contrarily corresponded, which concurs with our theory, that is the more the advance sum, the lower the APR.
#The advance unique sum is decidedly connected with the expressed month to month salary, it bodes well since borrowers with 
#all the more month to month pay could advance more mony.


# In[20]:


#Let's move on to looking at how borrower APR, stated monthly income and loan original 
#amount correlate with the categorical variables.


# In[21]:



# plot matrix of numeric features against categorical features.

def boxgrid(x, y, **kwargs):
    """ Quick hack for creating box plots with seaborn's PairGrid. """
    default_color = sb.color_palette()[0]
    sb.boxplot(x, y, color = default_color)

plt.figure(figsize = [10, 10])
g = sb.PairGrid(data = loan_sub, y_vars = ['BorrowerAPR', 'StatedMonthlyIncome', 'LoanOriginalAmount'], 
                x_vars = cat_vars, size = 5, aspect = 3.5)
g.map(boxgrid);
plt.xticks(rotation=30);


# In[22]:



#let's look at relationships between the three categorical features.


# In[55]:


plt.figure(figsize = [7, 10])
# Prosper rating vs term

plt.subplot(5, 1, 1)
sb.countplot(data = loan_sub, x = 'ProsperRating (Alpha)', hue = 'Term', palette = 'Blues')

#  employment status vs. term
ax = plt.subplot(3, 1, 2)
sb.countplot(data = loan_sub, x = 'EmploymentStatus', hue = 'Term', palette = 'Blues')
plt.xticks(rotation=10)

#  Prosper rating vs. employment status, use different color palette
ax = plt.subplot(3, 1, 3)
sb.countplot(data = loan_sub, x = 'EmploymentStatus', hue = 'ProsperRating (Alpha)', palette = 'Greens')
ax.legend(loc = 2, ncol = 2);
plt.xticks(rotation=10);


# In[24]:


#he employment status variable do not have enough data on part-time, retired and not employed borrowers to show its interaction
#with term and Prosper rating variables. But we can see that there is a interaction between term and Prosper rating. 
#Proportionally, there are more 60 month loans on B and C ratings. There is only 36 months loans for HR rating borrowers.

#With the preliminary look at bivariate relationships out of the way, I want to see how borrower APR and loan original amount 
#are related to one another for all of the data.


# In[25]:


plt.figure(figsize = [8, 6])

sb.regplot(data = loan_sub, x = 'LoanOriginalAmount', y = 'BorrowerAPR', scatter_kws={'alpha':0.01});


# In[26]:


#This plot shows that at different size of the loan amount, the APR has a large range, but the range of APR decrease
# with the increase of loan amount. Overall, the borrower APR is negatively correlated with loan amount.


# In[ ]:





# In[ ]:





# In[ ]:




