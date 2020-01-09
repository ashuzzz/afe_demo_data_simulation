import pandas as pd
import numpy as np
import random
import datetime 
import math
from scipy.stats import skewnorm

mySeed = 20171230
np.random.seed(mySeed)

# This code's objective is to simulate bureau data for building a project to demonstrate the Automated Feature Engineering capabilities of DataRobot
# We will create fake Credit Bureau Data - Tradeline.csv, PublicRecords.csv, Inquiries.csv, Collections.csv, and LoanApplications.csv

def sim_date(oldest_dt, duration, num_days):
  oldest_dt = datetime.datetime.strptime(oldest_dt,'%Y-%m-%d')
  dt_list = []
  for i in range(num_days):
    dt_list.append(oldest_dt + datetime.timedelta(days=int(np.random.choice(list(range(0,duration)),1))))
  return pd.Series(dt_list)

############################# Loan Applications Data ######################################

# This is the main file with which DataRobot project will be created
# Imperial Bank had 10,000 applications for Instalment loans between the dates of 2014-07-01 and 2014-09-30
# All these loans were approved and went through the entire loan cycle. Now, we know who defaulted(target=1) and paid back in full(target=0)
# Below, we simulate the information available at the time of making the lending decision

def simulate_loan_apps(customers):
  print("Simulating loan applications")
  nRows = len(customers)
  loans = pd.DataFrame(customers,columns=['CustID']).sort_values(by='CustID')

  #Application date
  oldest_app_date = '2014-07-01'
  app_window = 90 #3 months of applications
  loans["app_date"] = sim_date(oldest_app_date,app_window,nRows)
  loans["is_bad"] = np.random.choice([0,1],nRows,p=[0.9,0.1])

  #Annual Income
  loans["annual_income"] = np.random.binomial(1, 0.995, nRows)*np.round(np.random.gamma(1.7, 3, nRows)*10000, decimals=-2).astype('int64')
  loans.loc[loans["annual_income"]<1000,"annual_income"] = 1000
  
  #Marital Status
  mar_list=["single","married","divorced","widowed"]
  mar_prob_low = [0.6,0.15,0.23,0.02]
  mar_prob_med = [0.4,0.35,0.15,0.1]
  mar_prob_high = [0.15,0.6,0.05,0.2]
  loans.loc[loans["annual_income"]<30000,"marital_status"] = np.random.choice(mar_list,len(loans[loans["annual_income"]<30000]),p=mar_prob_low)
  loans.loc[(loans["annual_income"]>=30000) & (loans["annual_income"]<50000),"marital_status"] = np.random.choice(mar_list,len(loans[(loans["annual_income"]>=30000) & (loans["annual_income"]<50000)]),p=mar_prob_med)
  loans.loc[loans["annual_income"]>=50000,"marital_status"] = np.random.choice(mar_list,len(loans[loans["annual_income"]>=50000]),p=mar_prob_high)
  
  #Residential Status
  res_list=["rent","house_owner","mortgage"]
  res_prob_single=[0.85,0.05,0.1]
  res_prob_married=[0.3,0.2,0.5]
  res_prob_divorced=[0.6,0.1,0.3]
  res_prob_widowed=[0.1,0.6,0.3]
  loans.loc[loans["marital_status"]=="single","residential_status"] = np.random.choice(res_list,len(loans[loans["marital_status"]=="single"]),p=res_prob_single)
  loans.loc[loans["marital_status"]=="married","residential_status"] = np.random.choice(res_list,len(loans[loans["marital_status"]=="married"]),p=res_prob_married)
  loans.loc[loans["marital_status"]=="divorced","residential_status"] = np.random.choice(res_list,len(loans[loans["marital_status"]=="divorced"]),p=res_prob_divorced)
  loans.loc[loans["marital_status"]=="widowed","residential_status"] = np.random.choice(res_list,len(loans[loans["marital_status"]=="widowed"]),p=res_prob_widowed)
  
  #Loan purpose
  purpose_list=["auto","education","personal loan","business","debt_consolidation"]
  purpose_prob = [0.17,0.1,0.35,0.25,0.13]
  loans["loan_purpose"] = np.random.choice(purpose_list,nRows,p=purpose_prob)
  
  #Application Scores
  l=[]
  for index,app in loans.iterrows():
    dict_app = {}
    dict_app["CustID"] = loans["CustID"]

    #Annual Income
    if app["annual_income"]<20000:
      dict_app["inc_score"] = random.uniform(14,15)
    elif 20000<=app["annual_income"]<30000:
      dict_app["inc_score"] = random.uniform(11,12)
    elif 30000<=app["annual_income"]<40000:
      dict_app["inc_score"] = random.uniform(8,10)
    elif 40000<=app["annual_income"]<50000:
      dict_app["inc_score"] = random.uniform(6,7)
    elif 50000<=app["annual_income"]<60000:
      dict_app["inc_score"] = random.uniform(5,6)
    elif 60000<=app["annual_income"]<70000:
      dict_app["inc_score"] = random.uniform(4,5)
    elif 70000<=app["annual_income"]<80000:
      dict_app["inc_score"] = random.uniform(3,4)
    elif 80000<=app["annual_income"]<90000:
      dict_app["inc_score"] = random.uniform(1,2)
    else:
      dict_app["inc_score"] = random.uniform(0,0.5)
    
    ##Marital Status
    if app["marital_status"] == "single":
      dict_app["mar_score"] = random.uniform(2,6)
    elif app["marital_status"] == "married":
      dict_app["mar_score"] = random.uniform(1,3)
    elif app["marital_status"] == "divorced":
      dict_app["mar_score"] = random.uniform(5,9)
    elif app["marital_status"] == "widowed":
      dict_app["mar_score"] = random.uniform(0,4)

    #Residential Status
    if app["residential_status"] == "rent":
      dict_app["res_score"] = random.uniform(4,9)
    elif app["residential_status"] == "mortgage":
      dict_app["res_score"] = random.uniform(2,7)
    elif app["residential_status"] == "house_owner":
      dict_app["res_score"] = random.uniform(0,5)

    #Loan purpose
    if app["loan_purpose"] == "education":
      dict_app["purpose_score"] = random.uniform(0,1)
    elif app["loan_purpose"] == "debt_consolidation":
      dict_app["purpose_score"] = random.uniform(5,7)
    elif app["loan_purpose"] == "auto":
      dict_app["purpose_score"] = random.uniform(2,4)
    elif app["loan_purpose"] == "personal":
      dict_app["purpose_score"] = random.uniform(8,10)
    else:
      dict_app["purpose_score"] = random.uniform(12,15)

    #Application Variable weights
    a1= 0.45#annual income
    a2= 0.0#marital status
    a3= 0.1#residential status
    a4= 0.35#loan purpose

    dict_app["app_risk_score"] = a1*dict_app["inc_score"]+a2*dict_app["mar_score"]+a3*dict_app["res_score"]+a4*dict_app["purpose_score"]
    l.append(dict_app)

  scores = pd.DataFrame(l)
  max_score = max(scores["app_risk_score"]) 
  min_score = min(scores["app_risk_score"])
  loans["app_risk_score"] = scores["app_risk_score"].apply(lambda x: (x-min_score)/(max_score-min_score))
  loans["inc_score"] = scores["inc_score"]
  loans["purpose_score"] = scores["purpose_score"]
  loans["res_score"] = scores["res_score"]

  return loans
  

def agg_tradeline(tl):
  print("Aggregating tradeline table")
  rows = tl[["CustID","account_id"]].groupby("CustID").count().reset_index()
  rows.columns = ["CustID","num_tl_accounts"]

  #Account type
  rows["account_type_mostfreq"] = tl[['CustID','account_type']].groupby("CustID").agg(lambda x: x.value_counts().index[0]).reset_index()["account_type"]
  rows["account_type_numunique"] = tl[['CustID','account_type']].groupby("CustID").agg(lambda x: len(x.unique())).reset_index()["account_type"]

  #Creditor
  rows["creditor_mostfreq"] = tl[['CustID','creditor']].groupby("CustID").agg(lambda x: x.value_counts().index[0]).reset_index()["creditor"]
  rows["creditor_numunique"] = tl[['CustID','creditor']].groupby("CustID").agg(lambda x: len(x.unique())).reset_index()["creditor"]

  #Account owner
  rows["account_owner_mostfreq"] = tl[['CustID','account_owner']].groupby("CustID").agg(lambda x: x.value_counts().index[0]).reset_index()["account_owner"]
  rows["account_owner_numunique"] = tl[['CustID','account_owner']].groupby("CustID").agg(lambda x: len(x.unique())).reset_index()["account_owner"]

  #Current delinquency
  rows["curr_delq_mostfreq"] = tl[['CustID','current_delq']].groupby("CustID").agg(lambda x: x.value_counts().index[0] if len(x.value_counts())>0 else "NA").reset_index()["current_delq"]
  
  #Worst delq
  rows["worst_delq_mostfreq"] = tl[['CustID','worst_dlq']].groupby("CustID").agg(lambda x: x.value_counts().index[0] if len(x.value_counts())>0 else "NA").reset_index()["worst_dlq"]

  #Utilization
  rows["util_avg"]=tl[['CustID','utilization']].groupby("CustID").agg({'utilization':lambda x:x.mean()}).reset_index()["utilization"]

  #Credit limit
  rows["credit_limit_avg"] = tl[['CustID','credit_limit']].groupby("CustID").agg({'credit_limit':lambda x:x.mean()}).reset_index()["credit_limit"] 
  
  l=[]
  #Create scores for each aggregated variable
  for index,row in rows.iterrows():
    dict_tl = {}
    dict_tl["CustID"] = row["CustID"]

    #Number of tl accounts
    if row["num_tl_accounts"]==0:
      dict_tl["num_tl_score"] = random.uniform(6,10)
    elif 0<row["num_tl_accounts"]<3:
      dict_tl["num_tl_score"] = random.uniform(0,4)
    elif 3<=row["num_tl_accounts"]<5:
      dict_tl["num_tl_score"] = random.uniform(2,5)
    else:
      dict_tl["num_tl_score"] = random.uniform(3,6)

    #Most frequent account type
    if row["account_type_mostfreq"]=="revolving":
      dict_tl["account_type_mostfreq_score"] = random.uniform(2,7)
    elif row["account_type_mostfreq"]=="mortgage":
      dict_tl["account_type_mostfreq_score"] = random.uniform(3,5)
    else:
      dict_tl["account_type_mostfreq_score"] = random.uniform(1,4)
    
    #Num unique creditor
    if row["creditor_numunique"]==1:
      dict_tl["creditor_numunique_score"] = random.uniform(0,3)
    elif row["creditor_numunique"]==2:
      dict_tl["creditor_numunique_score"] = random.uniform(1,4)
    elif row["creditor_numunique"]==3:
      dict_tl["creditor_numunique_score"] = random.uniform(3,7)
    else:
      dict_tl["creditor_numunique_score"] = random.uniform(4,9)

    #Most freq creditor
    if row["creditor_mostfreq"] in ["ABC Bank","Bank of XYZ","Cooperative Capital"]:
      dict_tl["creditor_mostfreq_score"] = random.uniform(1,6)
    else:
      dict_tl["creditor_mostfreq_score"] = random.uniform(5,10)

    #Credit limit
    if row["credit_limit_avg"]==None:
      dict_tl["credit_limit_avg_score"] = random.uniform(0,4)
    elif row["credit_limit_avg"]<3000:
      dict_tl["credit_limit_avg_score"] = random.uniform(4,9)
    elif 3000<=row["credit_limit_avg"]<4000:
      dict_tl["credit_limit_avg_score"] = random.uniform(3,6)
    elif 4000<=row["credit_limit_avg"]<5000:
      dict_tl["credit_limit_avg_score"] = random.uniform(1,5)
    else:
      dict_tl["credit_limit_avg_score"] = random.uniform(0,3)

    # #Utilization
    # if row["util_avg"]==None:
    #   dict_tl["average_util_score"] = random.uniform(0,4)
    # elif row["util_avg"]<0.3 :
    #   dict_tl["average_util_score"] = random.uniform(1,4)
    # elif 0.3<=row["util_avg"]<0.5 :
    #   dict_tl["average_util_score"] = random.uniform(3,6)
    # elif 0.5<=row["util_avg"]<0.75 :
    #   dict_tl["average_util_score"] = random.uniform(5,7)
    # else:
    #   dict_tl["average_util_score"] = random.uniform(7,9)
    dict_tl["average_util_score"] = row["util_avg"]*10

    #Curr delinquency
    if row["curr_delq_mostfreq"]=="<30DPD":
      dict_tl["curr_delq_mostfreq_score"] = random.uniform(0,3)
    elif row["curr_delq_mostfreq"]=="30-60DPD":
      dict_tl["curr_delq_mostfreq_score"] = random.uniform(1,5)
    elif row["curr_delq_mostfreq"]=="60-90DPD":
      dict_tl["curr_delq_mostfreq_score"] = random.uniform(3,6)
    elif row["curr_delq_mostfreq"]==">90DPD":
      dict_tl["curr_delq_mostfreq_score"] = random.uniform(5,9)
    else:
      dict_tl["curr_delq_mostfreq_score"] = random.uniform(1,5)

    #Worst delinquency
    if row["worst_delq_mostfreq"]=="<30DPD":
      dict_tl["worst_delq_mostfreq_score"] = random.uniform(0,3)
    elif row["worst_delq_mostfreq"]=="30-60DPD":
      dict_tl["worst_delq_mostfreq_score"] = random.uniform(1,5)
    elif row["worst_delq_mostfreq"]=="60-90DPD":
      dict_tl["worst_delq_mostfreq_score"] = random.uniform(3,6)
    elif row["worst_delq_mostfreq"]==">90DPD":
      dict_tl["worst_delq_mostfreq_score"] = random.uniform(5,9)
    else:
      dict_tl["worst_delq_mostfreq_score"] = random.uniform(1,5)

    l.append(dict_tl)

  scores = pd.DataFrame(l)
  #TL variable weights
  t1 = 0.2 #num_tl
  t2 = 0.25 #curr_delq_mostfreq
  t3 = 0.0 #worst_delq_mostfreq
  t4 = 0.1 #creditor_numunique
  t5 = 0.05 #acct_type_mostfreq
  t6 = 0.3 #avg utilization
  t7 = 0.05 #avg credit limit
  t8 = 0.05 #most freq creditor

  scores["tl_risk_score"] = t1*scores["num_tl_score"] + t2*scores["curr_delq_mostfreq_score"] + t3*scores["worst_delq_mostfreq_score"] 
  + t4*scores["creditor_numunique_score"] + t5*scores["account_type_mostfreq_score"] + t6*scores["average_util_score"] + t7*scores["credit_limit_avg_score"]
  + t8*scores["creditor_mostfreq_score"]
  max_score = max(scores["tl_risk_score"]) 
  min_score = min(scores["tl_risk_score"])
  scores["tl_risk_score"] = scores["tl_risk_score"].apply(lambda x: (x-min_score)/(max_score-min_score))
  return rows.merge(scores,on="CustID")

def simulate_tradeline(customers,loans):
  print("Simulating tradeline accounts")
  nTradeLines = len(customers) * 3 #assuming that bureau contains 3 tradelines on average per customer
  tradeline = pd.DataFrame(np.random.choice(customers, nTradeLines, replace=True), columns=['CustID']).sort_values(by="CustID")

  #AccountID: Existing tradelines for applicant
  tradeline['account_id'] = ["A"+str(s) for s in np.random.choice(999999, size=nTradeLines,replace=False)] 

  #Account Type
  acct_type = ["revolving","mortgage","instalment"]
  acct_type_prob = [0.75,0.1,0.15]
  tradeline["account_type"] = np.random.choice(acct_type,nTradeLines,p=acct_type_prob)

  #Creditor
  list_banks = ["TrendingClub","ABC Bank","Bank of XYZ","Cooperative Capital","Rhyme","Lord_P2P","Uprise"]
  mkt_share = [0.1,0.34,0.21,0.1,0.09,0.07,0.09]
  tradeline["creditor"] = np.random.choice(list_banks,nTradeLines,p=mkt_share)

  #AccountOwner
  owner_list = ["individual","joint"]
  owner_prob = [0.9,0.1]
  tradeline["account_owner"] = np.random.choice(owner_list,nTradeLines,p=owner_prob)

  #Int_Rate
  tradeline["int_rate"] = None
  tradeline.loc[tradeline["account_type"]=="instalment","int_rate"] = np.random.normal(0.08,0.0016,len(tradeline[tradeline["account_type"]=="instalment"]))
  tradeline.loc[tradeline["account_type"]=="mortgage","int_rate"] = np.random.normal(0.06,.0009,len(tradeline[tradeline["account_type"]=="mortgage"]))
  tradeline.loc[tradeline["account_type"]=="revolving","int_rate"] = np.random.normal(0.09,0.0036,len(tradeline[tradeline["account_type"]=="revolving"]))

  #Credit limit
  tradeline["credit_limit"]=None
  tradeline.loc[tradeline["account_type"]=="revolving","credit_limit"] = np.random.randint(1000,8000,size=len(tradeline[tradeline["account_type"]=="revolving"]))
  tradeline["credit_limit"] = tradeline["credit_limit"].apply(lambda x: int(math.ceil(x)/100.0)*100 if x!=None else x)

  #Balance
  tradeline["balance"]=None
  tradeline["utilization"] = None
  tradeline.loc[tradeline["account_type"]=="revolving","utilization"] = list(np.clip(np.random.normal(0.5,0.3,len(tradeline[tradeline["account_type"]=="revolving"])),0,0.95))
  tradeline["balance"] = tradeline["credit_limit"]*tradeline["utilization"]

  #Open and close dates
  open_dates=[]
  close_dates = []
  mg_loan_life = np.random.normal(30*365,5*365,nTradeLines) #mortgage account could live between 6mo to 30 years
  other_loan_life = np.random.normal(1,60, nTradeLines) #revolving and instalment are for 0 to 5 years

  for i in range(len(tradeline)):
  #mortgage accounts opened between 2004-01-01 and 2014-07-01
    if (tradeline.iloc[i]["account_type"]=="mortgage"):
      open_dt = (datetime.date(2004,6,1)+datetime.timedelta(days = int(np.random.randint(0,3600,1)[0])))
      open_dates.append(open_dt)
      mg_days = int(np.random.normal(20*365,7*365,1)[0])
      close_dt = open_dt + datetime.timedelta(days = int(mg_days))
      if close_dt>datetime.date(2014,7,1) or close_dt<open_dt:
        close_dt=None
    else: #revolving and instalment loans captured for last 2 years
      open_dt = (datetime.date(2012,6,1)+datetime.timedelta(days = int(np.random.randint(0,365*2,1)[0])))
      open_dates.append(open_dt)
      other_days = np.random.normal(3*365,365,1)
      close_dt = open_dates[i] + datetime.timedelta(days = int(other_days))
      if close_dt>=datetime.date(2014,7,1) or close_dt<open_dt:
        close_dt=None
    close_dates.append(close_dt)

  tradeline["open_date"] = open_dates
  tradeline["closed_date"] = close_dates
  tradeline["report_date"] = datetime.date(2004,5,31)

  #Current delinquency status
  curr_status = ["<30DPD","30-60DPD","60-90DPD",">90DPD"]
  curr_status_prob = [0.8,0.08,0.07,0.05]
  tradeline["current_delq"] = np.random.choice(curr_status,nTradeLines,p=curr_status_prob) 

  #Worst Dlq in last 12 months
  tradeline.loc[tradeline["current_delq"].isnull(),"worst_dlq"] = np.random.choice(["<30DPD","30-60DPD","60-90DPD",">90DPD"],len(tradeline[tradeline["current_delq"].isnull()]),p=[0.9,0.05,0.03,0.02]) #for closed accounts
  tradeline.loc[tradeline["current_delq"]=="<30DPD","worst_dlq"]=np.random.choice(["<30DPD","30-60DPD","60-90DPD",">90DPD"],len(tradeline[tradeline["current_delq"]=="<30DPD"]),p=[0.9,0.05,0.03,0.02]) #accounts that are current as of now
  tradeline.loc[tradeline["current_delq"]=="30-60DPD","worst_dlq"]=np.random.choice(["30-60DPD","60-90DPD",">90DPD"],len(tradeline[tradeline["current_delq"]=="30-60DPD"]),p=[0.9,0.06,0.04]) #accounts currently 1 cycle delq
  tradeline.loc[tradeline["current_delq"]=="60-90DPD","worst_dlq"]=np.random.choice(["60-90DPD",">90DPD"],len(tradeline[tradeline["current_delq"]=="60-90DPD"]),p=[0.8,0.2]) #accounts currently 2 cycles delq
  tradeline.loc[tradeline["current_delq"]==">90DPD","worst_dlq"]=">90DPD" #accounts currently 3 cycles delq
  tradeline.loc[tradeline["closed_date"]<datetime.date(2013,7,1),"worst_dlq"]=None #for accounts closed before report_date-12 months
  
  #When account is closed
  tradeline.loc[~tradeline["closed_date"].isnull(),"balance"] = None
  tradeline.loc[~tradeline["closed_date"].isnull(),"credit_limit"] = None
  tradeline.loc[~tradeline["closed_date"].isnull(),"utilization"] = None
  tradeline.loc[~tradeline["closed_date"].isnull(),"current_delq"]=None #for accounts closed before report_date

  tl_agg = agg_tradeline(tradeline)
  tl_agg.to_csv("agg_tl.csv",index=False, header=True)

  tradeline = tradeline.merge(tl_agg[["CustID","tl_risk_score"]],on="CustID")
  return tradeline

def agg_inq(inq):
  print("Aggregating inquiries table")
  rows = inq[["CustID","inquiry_id"]].groupby("CustID").count().reset_index().sort_values(by="CustID")
  rows.columns=["CustID","num_inquiries"]

  #Inquiry type
  rows["inq_type_mostfreq"] = inq[['CustID','inquiry_type']].groupby("CustID").agg(lambda x: x.value_counts().index[0]).reset_index()["inquiry_type"]
  rows["inq_type_numunique"] = inq[['CustID','inquiry_type']].groupby("CustID").agg(lambda x: len(x.unique())).reset_index()["inquiry_type"]

  #Application decision
  rows["application_decision_mostfreq"] = inq[['CustID','application_decision']].groupby("CustID").agg(lambda x: x.value_counts().index[0]).reset_index()["application_decision"]
  rows["application_decision_numunique"] = inq[['CustID','application_decision']].groupby("CustID").agg(lambda x: len(x.unique())).reset_index()["application_decision"]

  #Inquiry Scores
  l=[]

  for index, inq in rows.iterrows():
    dict_inq={}
    dict_inq["CustID"] = rows["CustID"]

    #Number of inquiries
    if inq["num_inquiries"]==None or inq["num_inquiries"]==0:
      dict_inq["num_inq_score"] = random.uniform(0,3)
    elif inq["num_inquiries"]==1:
      dict_inq["num_inq_score"] = random.uniform(0,5)
    elif 1<inq["num_inquiries"]<4:
      dict_inq["num_inq_score"] = random.uniform(1,6)
    elif inq["num_inquiries"]>4:
      dict_inq["num_inq_score"] = random.uniform(3,8)
    else:
      dict_inq["num_inq_score"] = random.uniform(0,4)

    #app_dec_mostfreq
    if inq["application_decision_mostfreq"]=="approved":
      dict_inq["appdec_mf_score"] = random.uniform(0,5)
    else:
      dict_inq["appdec_mf_score"] = random.uniform(3,8)

    #Inq type num unique
    if inq["inq_type_numunique"]==None or inq["inq_type_numunique"]==0:
      dict_inq["inq_type_numunique_score"] = random.uniform(0,4)
    elif 1<inq["inq_type_numunique"]<4:
      dict_inq["inq_type_numunique_score"] = random.uniform(2,7)
    elif inq["inq_type_numunique"]>=4:
      dict_inq["inq_type_numunique_score"] = random.uniform(5,9)

    #App decision num unique
    if inq["application_decision_numunique"]==None or inq["application_decision_numunique"]==0:
      dict_inq["application_decision_numunique_score"] = random.uniform(0,4)
    elif inq["application_decision_numunique"]==1:
      dict_inq["application_decision_numunique_score"] = random.uniform(0,5)
    else:
      dict_inq["application_decision_numunique_score"] = random.uniform(3,7)

    #inq_type_mostfreq
    if inq["inq_type_mostfreq"]=="revolving":
      dict_inq["inq_type_mostfreq_score"] = random.uniform(3,8)
    elif inq["inq_type_mostfreq"]=="instalment":
      dict_inq["inq_type_mostfreq_score"] = random.uniform(1,5)
    elif inq["inq_type_mostfreq"]=="mortgage":
      dict_inq["inq_type_mostfreq_score"] = random.uniform(2,7)
    elif inq["inq_type_mostfreq"]=="rental_application":
      dict_inq["inq_type_mostfreq_score"] = random.uniform(0,4)
    else:
      dict_inq["inq_type_mostfreq_score"] = random.uniform(2,5)

    l.append(dict_inq)

  scores=pd.DataFrame(l)
  #final score, weights for each aggregated feature
  b1 = 0.3 #num_inq
  b2 = 0.25 #inq_type_mf
  b3 = 0.1 #inqtype_numunique
  b4 = 0.3 #appdec_mf
  b5 = 0.05 #appdec_numunique
  
  scores["inq_risk_score"] = b1*scores["num_inq_score"]+b2*scores["inq_type_mostfreq_score"]+b3*scores["inq_type_numunique_score"]+b4*scores["appdec_mf_score"]+b5*scores["application_decision_numunique_score"]
  max_score = max(scores["inq_risk_score"]) 
  min_score = min(scores["inq_risk_score"])
  scores["inq_risk_score"] = scores["inq_risk_score"].apply(lambda x: (x-min_score)/(max_score-min_score))
  rows["inq_risk_score"] = scores["inq_risk_score"]
  return rows

def simulate_inq(customers,loans):
  print("Simulating inquiries table")
  nInq = len(customers) * 3 #assuming that bureau contains 3 inquiries on average per customer
  inq = pd.DataFrame(np.random.choice(customers, nInq, replace=True), columns=['CustID']).sort_values(by="CustID")
  inq["inquiry_id"] = ["Inq_"+str(s) for s in np.random.choice(range(100000, 999999), size=nInq, replace=False)]

  inq = inq.merge(loans[["CustID","app_date"]].groupby("CustID").min(),on="CustID",how="left")
  # inq.to_csv("inq_test.csv")

  #Date of inquiry
  inq["inquiry_date"] = inq["app_date"].apply(lambda x: x-datetime.timedelta(int(np.random.randint(0,120,1)[0])))
  inq = inq.drop(["app_date"],axis=1)

  #InquiryType
  inq_type = ["revolving","instalment","mortgage","rental_application"]
  inq_type_prob = [0.65,0.1,0.1,0.15]
  inq["inquiry_type"] = np.random.choice(inq_type,nInq,p=inq_type_prob)

  inq["report_date"] = datetime.date(2014,5,31)
  #ApplicationStatus
  app_status = ["approved","denied"]
  app_status_prob = [0.8,0.2]
  inq["application_decision"] = np.random.choice(app_status,nInq,p=app_status_prob)

  inq_agg = agg_inq(inq)
  inq_agg.to_csv("inq_agg.csv")

  inq = inq.merge(inq_agg[["CustID","inq_risk_score"]],on="CustID")
  return inq

def main():
  print("Simulating credit bureau data...")
  nRows = 10000
  customers = ["C"+str(s) for s in np.random.choice(range(100000, 999999), size=nRows, replace=False)]
  
  loans = simulate_loan_apps(customers)
  tradeline = simulate_tradeline(customers,loans)
  inq = simulate_inq(customers,loans)

  #Fix target variable
  print("Creating target variable")

  #Merge all leaf risk scores with root table
  loans = loans.merge(inq[["CustID","inq_risk_score"]].groupby("CustID").mean(), how="left", on="CustID")
  loans = loans.merge(tradeline[["CustID","tl_risk_score"]].groupby("CustID").mean(), how="left", on="CustID")
  # loans = loans.merge(pr[["CustID","pr_risk_score"]].groupby("CustID").mean(), how="left", on="CustID")
  
  #Combine all leaf risk scores
  #Leaf weights
  # residential_wt = 0.05
  # annual_income_wt = 0.2
  # loan_purpose_wt = 0.2
  app_wt=0.6
  tl_wt = 0.3
  inq_wt = 0.1
  # pr_wt = 0.2

  #Inq_score missing
  loans.loc[loans["inq_risk_score"].isnull(),"inq_risk_score"] = loans["inq_risk_score"].quantile(0.1)
  #TL_Score missing
  loans.loc[loans["tl_risk_score"].isnull(),"tl_risk_score"] = loans["tl_risk_score"].quantile(0.5)
  #Both missing

  loans["final_score"] = app_wt*loans["app_risk_score"]+tl_wt*loans["tl_risk_score"]+inq_wt*loans["inq_risk_score"]
  # loans["final_score"] = annual_income_wt*loans["inc_score"]+ loan_purpose_wt*loans["purpose_score"] + tl_wt*loans["tl_risk_score"]+inq_wt*loans["inq_risk_score"]

  thresh = loans["final_score"].quantile(0.9) #top 10% will have is_bad=1
  loans["is_bad"] = loans["final_score"].apply(lambda x: 0 if x==None else (1 if x>thresh else 0))
  loans.loc[(loans["annual_income"]>104000) & (loans["annual_income"]<114000),"is_bad"] = np.random.choice([0,1],len(loans[(loans["annual_income"]>104000) & (loans["annual_income"]<114000)]),p=[0.94,0.06])
  loans.loc[(loans["annual_income"]>115000) & (loans["annual_income"]<125000),"is_bad"] = np.random.choice([0,1],len(loans[(loans["annual_income"]>115000) & (loans["annual_income"]<125000)]),p=[0.96,0.04])
  loans.loc[(loans["annual_income"]>125000) & (loans["annual_income"]<135000),"is_bad"] = np.random.choice([0,1],len(loans[(loans["annual_income"]>125000) & (loans["annual_income"]<135000)]),p=[0.98,0.02])
  loans.loc[loans["annual_income"]>135000,"is_bad"] = np.random.choice([0,1],len(loans[loans["annual_income"]>135000]),p=[0.99,0.01])
  
  loans.loc[loans["loan_purpose"]=="business","is_bad"] = np.random.choice([0,1],len(loans[loans["loan_purpose"]=="business"]),p=[0.85,0.15])
  loans.loc[loans["loan_purpose"]=="education","is_bad"] = np.random.choice([0,1],len(loans[loans["loan_purpose"]=="education"]),p=[0.95,0.05])
  
  loans.loc[loans["marital_status"]=="widowed","is_bad"] = np.random.choice([0,1],len(loans[loans["marital_status"]=="widowed"]),p=[0.93,0.07])
  loans.loc[loans["marital_status"]=="divorced","is_bad"] = np.random.choice([0,1],len(loans[loans["marital_status"]=="divorced"]),p=[0.88,0.12])
  
  #Drop latent factors
  loans = loans.drop(["app_risk_score","inq_risk_score","tl_risk_score","final_score","inc_score","purpose_score","res_score"],axis=1)
  tradeline = tradeline.drop(["tl_risk_score"],axis=1)
  inq = inq.drop(["inq_risk_score"],axis=1)
  # pr = pr.drop(["pr_risk_score"],axis=1)

  #Cosmetic changes
  # tradeline["int_rate"] = tradeline["int_rate"].apply(lambda x: str(round(100*x,2))+"%")
  # tradeline["credit_limit"] = tradeline["credit_limit"].apply(lambda x: "$"+str(int(x/100)*100) if x!=None else None)
  # tradeline["balance"] = tradeline["balance"].apply(lambda x: "$"+str(int(x/100)*100)

  #Writing out csv files
  print("Writing out files")
  loans.to_csv('Loan Applications.csv',index=False, header=True)
  inq.to_csv("Bureau Inquiries.csv",index=False, header=True)
  tradeline.to_csv("Bureau Tradeline Accounts.csv",index=False, header=True)
  # pr.to_csv("Public Records.csv")

if __name__=='__main__':
  main()
  
