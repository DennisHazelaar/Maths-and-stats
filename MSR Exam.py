# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 15:30:42 2023

@author: denni
"""

#a - Read the data
import pandas as pd
import statistics
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('2123286alienation_data.csv')
print(df.head)

alienation = df.drop(['income','male','consult'], axis =1)

#b - summarize your sample of 100 respondents
mean = alienation.mean()
median = alienation.median()
alienation_range = alienation.max()-alienation.min()
std = alienation.std()

#histogram alienation
plt.style.use('ggplot')
plt.hist(alienation, bins=10)
plt.xlabel('alienation')
plt.ylabel('frequency')
plt.title('Alienation of all respondents')
plt.show()
#alienation is not normally distributed

#females
mean_f_alienation = df.loc[df['male'] == 0,'alienation'].mean()
median_f_alienation = df.loc[df['male'] == 0,'alienation'].median()
range_f_alienation = df.loc[df['male'] == 0,'alienation'].max() - df.loc[df['male'] == 0, 'alienation'].min()
std_f_alienation = df.loc[df['male'] == 0,'alienation'].std()
 
#create a df with only females
female_df = df[df['male'] == 0]

#create a histogram for female alienation
plt.style.use('ggplot')
plt.hist(female_df['alienation'], bins=10)
plt.xlabel('alienation')
plt.ylabel('frequency')
plt.title('Alienation of female respondents')
plt.show()
#female alienation is not normally distributed

#males
mean_m_alienation = df.loc[df['male'] == 1,'alienation'].mean()
median_m_alienation = df.loc[df['male'] == 1,'alienation'].median()
range_m_alienation = df.loc[df['male'] == 1,'alienation'].max() - df.loc[df['male'] == 1, 'alienation'].min()
std_m_alienation = df.loc[df['male'] == 1,'alienation'].std()
 
#create a df with only males
male_df = df[df['male'] == 1]

#create a histogram for male alienation
plt.style.use('ggplot')
plt.hist(male_df['alienation'], bins=10)
plt.xlabel('alienation')
plt.ylabel('frequency')
plt.title('Alienation of male respondents')
plt.show()
#male alienation is not normally distributed but is closer to normal distribution than the females and the average


#income
income = df.drop(['alienation', 'male', 'consult'], axis=1)

mean_income = income.mean()
median_income = income.median()
range_income = income.max() - income.min()
std_income = income.std()

#create a histogram for the income
plt.style.use('ggplot')
plt.hist(income, bins=10)
plt.xlabel('income')
plt.ylabel('frequency')
plt.title('income of all respondents')
plt.show()

#females
mean_f_income = df.loc[df['male'] == 0,'income'].mean()
median_f_income = df.loc[df['male'] == 0,'income'].median()
range_f_income = df.loc[df['male'] == 0,'income'].max() - df.loc[df['male'] == 0, 'income'].min()
std_f_income = df.loc[df['male'] == 0,'income'].std()
 
#create a df with only females
female_df_income = df[df['male'] == 0]

#create a histogram for female income
plt.style.use('ggplot')
plt.hist(female_df_income['income'], bins=10)
plt.xlabel('income')
plt.ylabel('frequency')
plt.title('income of female respondents')
plt.show()
#female income is not normally distributed and there are a lot of females with no income

#males
mean_m_income = df.loc[df['male'] == 1,'income'].mean()
median_m_income = df.loc[df['male'] == 1,'income'].median()
range_m_income = df.loc[df['male'] == 1,'income'].max() - df.loc[df['male'] == 1, 'income'].min()
std_m_income = df.loc[df['male'] == 1,'income'].std()
 
#create a df with only males
male_df_income = df[df['male'] == 1]

#create a histogram for male income
plt.style.use('ggplot')
plt.hist(male_df['income'], bins=10)
plt.xlabel('income')
plt.ylabel('frequency')
plt.title('income of male respondents')
plt.show()
#male income is not normally distributed and there are a lot of people who earn around 80k

#to find out what data is missing we are looking for isna
alienation_missing = df['alienation'].isna().sum()
income_missing = df['income'].isna().sum()
gender_missing = df['male'].isna().sum()
consult_missing = df['consult'].isna().sum()

#none of the data seems to be NA so now we are going to look at the dataset
#if there would be any data missing we would either delete the rows or create dummy variables

#how many people have sought help?
phelp = (df['consult'] == 1).sum()
print(phelp)
#9 people have sought psychological help

#venn diagram
from matplotlib_venn import venn2
from matplotlib_venn import venn3

set1 = set(df[df['male'] ==1].index)
set2 = set(df[df['male'] ==0].index)
set3 = set(df[df['consult'] ==1].index)

venn3([set1, set2, set3], set_colors=('skyblue', 'salmon','darkgreen'), set_labels=('Male', 'female', 'Consult'))

#12
import math
people = 9
group_size = 3
combinations = math.factorial(people) / (math.factorial(group_size) * math.factorial(people-group_size))
print(combinations)
#you can make 84 combinations

n_males = 2
n_females = 7

combinations1mexact = math.comb(n_males,1) * math.comb(n_females, 2)
print(combinations1mexact)
#you can make 42 combinations with atleast 1 male

combinations1matleast = math.comb(n_males, 1) * math.comb(n_females, 2) + math.comb(n_males, 2) * math.comb(n_females, 1)
print(combinations1matleast)
#you can make 49 combinations with atleast one male in the group

#exchange rate is 1 dollar = 0,92 euro
df_in_euros = df['income'].apply(lambda x: x * 0.92)
print(df_in_euros)
print(df)

mean_income_euros = df_in_euros.mean()
print(mean_income_euros)
#mean income is 52252,73 euros and in dollars this was 	56795.36 dollar

std_income_euros = df_in_euros.std()
print(std_income_euros)
print(std_income)
#the std income is 31403.98 euros and in dollars this was 34134.76 dollar

#centered income
dfcenter = df.copy()
dfcenter['income'] = df['income'] - 56795.36
print(dfcenter)
mean_income_center= dfcenter['income'].mean()
print(mean_income_center)
#mean income is -0,00
std_income_center = dfcenter['income'].mean()
print(std_income_center)
#std income is -0,00

from scipy.optimize import fsolve

def alienation1(income):
    return 5.56 - 0.073 * income

def alienation2(income):
    return 4.80 - 0.030 * income

def func_to_solve(x):
    return alienation1(x) - alienation2(x)

x0 = 0  # initial guess for the intersection
intersection = fsolve(func_to_solve, x0)
print(intersection)

#Check at what alienation level the income is 
5.56 - 0.073 * 17.67
