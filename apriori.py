#!/usr/bin/env python
# coding: utf-8

# In[24]:


#importing the necessary packages
import pandas as pd
import numpy as np
import time
import logging
import sys
import csv
from collections import defaultdict
from itertools import chain, combinations
#running apriori using inbuilt library
from apyori import apriori


# In[31]:


#now lets try implementing apriori from scratch and see the results 
def find_subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

#for implementing apriori from scratch
def apriori_generate_freqitemsets(transactions,min_sup,min_conf):
    #so now we have the transactions with us 
    #lets get the itemsets
    #contains all the unique items in the transactions 
    itemset=set()
    for i in transactions:
        for j in i:
            if j not in itemset:
                itemset.add(frozenset([j]))

    #converting the transactions also to frozenset so that the error : unhashable typeset is removed
    t=list()
    for i in transactions:
        t.append(frozenset(i))
    transactions=t
    #print("Number of transactions: ",len(transactions))
    #print("No of individual items present in the data :",len(itemset))
    #now lets find the frequent itemsets from these
    itemsets_count=defaultdict(int)
    freqitemsets=set()
    for item in itemset:
        for t in transactions:
            if item.issubset(t):
                itemsets_count[item]+=1
    #traverse over dict to find support and then see whether it is frequent or not
    lendata=len(transactions)
    for i,c in itemsets_count.items():
        supp=float(c)/lendata
        if supp >= min_sup:
            freqitemsets.add(i)
    #print("No of frequent items present in the data of length 1:",len(freqitemsets))
    #now we will start generating itemsets of size 2 , 3 , 4 so on untill we cannot generate any such itemsets
    #so we will have an empty set with which we will be comparing it !
    setempty=set([])
    current_freqitemset=freqitemsets
    #list offreqitemsets of various lengths
    listfreqitemsets=list()
    num=1
    while(current_freqitemset!=setempty):
        listfreqitemsets.append(current_freqitemset)
        num=num+1
        #joining these i.e. merging these sets to form item sets of size num
        temp_freq = list()
        for i in current_freqitemset:
            for j in current_freqitemset:
                temp=set(i).union(set(j))
                if len(temp) == num:
                    temp_freq.append(frozenset(temp))
        #we got a list of temp_freq now lets convert it into a set
        temp_freq=set(temp_freq)
        #now we have the freq itemsets of order num lets check for their support
        itemset_count=defaultdict(int)
        freqitemsets=set()
        for item in temp_freq:
            for t in transactions:
                if item.issubset(t) == False:
                    continue
                else:
                    #incrementing locally
                    itemset_count[item]=itemset_count[item]+1
                    #incrementing globally
                    itemsets_count[item]=itemsets_count[item]+1
        #traverse over dict to find support and then see whether it is frequent or not 
        #finding support
        for i,c in itemset_count.items():
            supp=float(c)/lendata
            if supp >= min_sup:
                freqitemsets.add(i)
        #so we got freqitemsets of the given length num
        #print("No of frequent items present in the data of length ",num,":",len(freqitemsets))
        current_freqitemset=freqitemsets
    #after this while loop we will get the frequent itemsets of required support of all the lengths
    #print(len(listfreqitemsets))
    #lets see the possible rules for these now 
    return listfreqitemsets,itemsets_count

def apriori_find_rules(lendata,listfreqitemsets,itemsets_count,min_conf):
    num_rules=0
    lhs_rules=list()
    rhs_rules=list()
    rules=[]
    for i in listfreqitemsets:
        #print(i)
        count_item=0
        for item in i:
            #print(item)
            count_item+=1
            subsets = map(frozenset, [x for x in find_subsets(item)])
            for j in subsets:
                remain = item.difference(j)
                len_remain=len(remain)
                #print(remain)
                #now lets see for the rule
                if len_remain == 0:
                    continue
                #lets find the confidence now
                num=float(itemsets_count[item])/lendata
                den=float(itemsets_count[j])/lendata
                if den!=0 :
                    confidence=num/den
                    if confidence >= min_conf:
                        p1=list(j)
                        p2=list(remain)
                        if p1 in lhs_rules:
                            ind=lhs_rules.index(p1)
                            if p2 != rhs_rules[ind]:
                                if p2 in lhs_rules:
                                    ind=lhs_rules.index(p2)
                                    if p1 != rhs_rules[ind]:
                                        lhs_rules.append(p1)
                                        rhs_rules.append(p2)
                                        num_rules+=1
                                        rules.append((p1,p2))
                        elif p2 in lhs_rules:
                            ind=lhs_rules.index(p2)
                            if p1 != rhs_rules[ind]:
                                lhs_rules.append(p1)
                                rhs_rules.append(p2)
                                num_rules+=1 
                                rules.append((p1,p2))
                        else:
                            lhs_rules.append(p1)
                            rhs_rules.append(p2)
                            num_rules+=1   
                            rules.append((p1,p2))
                #print(count_item)
    print(num_rules)
    return rules


# In[32]:


#now for the partitioning method we need to generate frequent itemsets using partition method and after that we can formulate the rules
#from each partition we have to find frequent itemsets given the support
def find_rules_partitioned(lendata,freqitemset,min_conf,min_sup):
    num_rules=0
    itemsets_count = dict()
    lhs_rules=list()
    freqitemsets = list()
    rhs_rules=list()
    rules=[]
    #lendata,global_freqset,final_items,0.5
    for partition in freqitemset:
        for item in partition:
            if item[0] not in itemsets_count.keys():
                itemsets_count[item[0]] =0
            itemsets_count[item[0]] += item[1]
    for i in itemsets_count.keys():
         if ( itemsets_count[i]/lendata ) >= min_sup:
            freqitemsets.append(i)
    for item in freqitemsets:
            #print(item)
            subsets = map(frozenset, [x for x in find_subsets(item)])
            for j in subsets:
                remain = item.difference(j)
                len_remain=len(remain)
                #print(remain)
                #now lets see for the rule
                if len_remain == 0:
                    continue
                #lets find the confidence now
                num=float(itemsets_count[item])/lendata
                den=float(itemsets_count[j])/lendata
                if den!=0 :
                    confidence=num/den
                    if confidence >= min_conf:
                        p1=list(j)
                        p2=list(remain)
                        if p1 in lhs_rules:
                            ind=lhs_rules.index(p1)
                            if p2 != rhs_rules[ind]:
                                if p2 in lhs_rules:
                                    ind=lhs_rules.index(p2)
                                    if p1 != rhs_rules[ind]:
                                        lhs_rules.append(p1)
                                        rhs_rules.append(p2)
                                        num_rules+=1
                                        rules.append((p1,p2))
                        elif p2 in lhs_rules:
                            ind=lhs_rules.index(p2)
                            if p1 != rhs_rules[ind]:
                                lhs_rules.append(p1)
                                rhs_rules.append(p2)
                                num_rules+=1 
                                rules.append((p1,p2))
                        else:
                            lhs_rules.append(p1)
                            rhs_rules.append(p2)
                            num_rules+=1   
                            rules.append((p1,p2))
                #print(count_item)
    print(num_rules)
    return rules
    
def apriori_improvised_partitions(transactions,num_partitions,min_sup,min_conf):
    #so we have the transactions with us so we will first of all create partitions
    lendata = len(transactions)#getting transactions
    #lets start partitioning
    parts=list()
    freqitemsets=list()
    start=0
    size_partition=int(lendata/num_partitions)
    #for storing the running times of each partitions
    time_list=list()
    #time required for partitioning
    start_t = time.time()
    for c in range(num_partitions-1):
        parts.append( transactions[start:start+size_partition] )
        start=start+size_partition
    #if we have some elements left we allocate it to the last partition
    if start<lendata:
        parts.append(transactions[start:])
    end_t = time.time()
    time_partitioning=end_t-start_t
    print("Time Required for partitioning :",time_partitioning)
    #lets see the no of elements in each partition
    c=1
    for i in parts:
        print("Length of partition ",c," :",len(i))
        c=c+1
    #so for each partition we will find the freqitemsets and we will calculate time for each partition as well
    c=1
    freqitemsets=list()
    for p in parts:
        s = time.time()
        itemsets,count=apriori_generate_freqitemsets(transactions,min_sup,min_conf)
        temp_items = list()
        for  value in itemsets:
            temp_items.extend([(item, count[item]) for item in value])
        freqitemsets.append(temp_items)
        en = time.time()
        partition_time=en-s
        time_list.append(partition_time)
        #print("Time required by partition number :",c," is:",partition_time)
        c=c+1
    return freqitemsets,time_partitioning,min(time_list)


# In[33]:


#method2
#transaction reduction
def apriori_improvised_transactionreduction(transactions,min_sup,min_conf):
    lendata=len(transactions)
    c1=defaultdict(int)
    freq_set=defaultdict(int)
    itemsets_count=defaultdict(int)
    #lets get the 1st level first
    level1=set()
    min_supp_count=int(min_sup*lendata)
    removed=set()
    tran_new=list()
    for t in transactions:
        for i in t:
            temp=frozenset([i])
            c1[temp] = c1[temp]+1
            freq_set[temp]=freq_set[temp]+1
    for i in c1.keys():
        val=c1[i]
        if val < min_supp_count:
            removed.add(i)
        else:
            level1.add(i)
    for i in transactions:
        temp=set(i)
        for j in removed:
            a, =j
            if a in temp:
                temp.remove(a)
        temp=list(temp)
        if len(temp) >= 2:
            tran_new.append(temp)
    transactions=tran_new#referring the new transactions
    setempty=set([])
    current_freqitemset=level1
    #list offreqitemsets of various lengths
    listfreqitemsets=list()
    num=0
    while(current_freqitemset!=setempty):
        listfreqitemsets.append(current_freqitemset)
        num=num+1
        #joining these i.e. merging these sets to form item sets of size num
        temp_freq = list()
        for i in current_freqitemset:
            for j in current_freqitemset:
                temp=set(i).union(set(j))
                if len(temp) == num:
                    temp_freq.append(frozenset(temp))
        #we got a list of temp_freq now lets convert it into a set
        temp_freq=set(temp_freq)
        #now we have the freq itemsets of order num lets check for their support
        itemset_count=defaultdict(int)
        freqitemsets=set()
        for item in temp_freq:
            for t in transactions:
                if item.issubset(t) == False:
                    continue
                else:
                    #incrementing locally
                    itemset_count[item]=itemset_count[item]+1
                    #incrementing globally
                    itemsets_count[item]=itemsets_count[item]+1
        #traverse over dict to find support and then see whether it is frequent or not 
        #finding support
        for i,c in itemset_count.items():
            supp=float(c)/lendata
            if supp >= min_sup:
                freqitemsets.add(i)
        #so we got freqitemsets of the given length num
        print("No of frequent items present in the data of length ",num,":",len(freqitemsets))
        current_freqitemset=freqitemsets
    num_rules=0
    lhs_rules=list()
    rhs_rules=list()
    rules=[]
    for i in listfreqitemsets:
        #print(i)
        count_item=0
        for item in i:
            #print(item)
            count_item+=1
            subsets = map(frozenset, [x for x in find_subsets(item)])
            for j in subsets:
                remain = item.difference(j)
                len_remain=len(remain)
                #print(remain)
                #now lets see for the rule
                if len_remain == 0:
                    continue
                #lets find the confidence now
                num=float(itemsets_count[item])/lendata
                den=float(itemsets_count[j])/lendata
                if den!=0 :
                    confidence=num/den
                    if confidence >= min_conf:
                        p1=list(j)
                        p2=list(remain)
                        if p1 in lhs_rules:
                            ind=lhs_rules.index(p1)
                            if p2 != rhs_rules[ind]:
                                if p2 in lhs_rules:
                                    ind=lhs_rules.index(p2)
                                    if p1 != rhs_rules[ind]:
                                        lhs_rules.append(p1)
                                        rhs_rules.append(p2)
                                        num_rules+=1
                                        rules.append((p1,p2))
                        elif p2 in lhs_rules:
                            ind=lhs_rules.index(p2)
                            if p1 != rhs_rules[ind]:
                                lhs_rules.append(p1)
                                rhs_rules.append(p2)
                                num_rules+=1 
                                rules.append((p1,p2))
                        else:
                            lhs_rules.append(p1)
                            rhs_rules.append(p2)
                            num_rules+=1   
                            rules.append((p1,p2))
                #print(count_item)
    print(num_rules)
    return rules


# In[41]:


def set_1(filename,min_sup,min_conf):
    file = open(filename, 'r') 
    data = file.readlines() 
    print("No of lines in data :",len(data))
    #now lets iterate over this data and append the transactions
    transactions=list()
    for i in data:
        #as data is separated by -1 lets use spliting on the basis of -1 
        line = i.split(' -1 ')
        #checking if last element is '-2\n' and removing it
        ind=len(line)-1
        if line[ind] == '-2\n':
            line.pop()
        #appending it to transactions
        transactions.append(line)
    print("No of transactions:",len(transactions))
    print("File Read Successfully ! Data Stored in list : transactions")
    start = time.time()
    print("Using inbuilt Apriori Algorithm")
    #we can input the values of support and confidence but here i am taking a support of 70% and confidence of 50%
    association_rules = apriori(transactions,min_support=min_sup,min_confidence=min_conf,min_lift=1)
    association_results = list(association_rules)
    num_rules=0
    for i in association_results:
        num_rules=num_rules+1
    print("No of rules:",num_rules)
    num_val=0
    for i in association_results:
        if(len(list(i[0]))>1 and str(list(i[0])[0])!='NaN' and str(list(i[0])[1])!='NaN'):
            num_val=num_val+1
            print("Rule Number "+str(num_val)+" :"+str(list(i[0])[0])+" -> "+str(list(i[0])[1]),end='')    
            for j in range(2,len(list(i[0])),1):
                if(str(list(i[0])[j])!='NaN'):
                    print(" & "+str(list(i[0])[j]),end='')
            print()
    print("No of valid rules: ",num_val)
    end = time.time()
    overall_time_inbuilt=end-start
    print("Overall time for inbuilt apriori:",overall_time_inbuilt)
    print("Using Apriori Developed from scratch")
    start = time.time()
    itemsets,itemsets_count=apriori_generate_freqitemsets(transactions,min_sup,min_conf)
    lendata=len(transactions)
    #finding rules
    association_rules=apriori_find_rules(lendata,itemsets,itemsets_count,min_conf)
    end = time.time()
    overall_time1=end-start
    print("Overall time for apriori implementation from scratch: ",overall_time1)
    print("Rules we got")
    c=1
    for i in association_rules:
        print("Rule Number ",c," :",i[0]," -> ",i[1])
        c=c+1
    print("Now lets improvise the Apriori Algorithm built from scratch:")
    print("-------------------Improvisation Number : 1---------------------------------")
    print("Partitioning ")
    #partitioning and taking time accordingly
    freqitemsets,t1,t2=apriori_improvised_partitions(transactions,5,min_sup,min_conf)
    #before finding the rules we need to find the count of elements as well
    s = time.time()
    association_rules=find_rules_partitioned(lendata,freqitemsets,min_conf,min_sup)
    e=time.time()
    #print(t1+t2)
    t3=e-s
    total_time_partition=t1+t2+t3
    print("Total time:",total_time_partition)
    print("-------------------Improvisation Number : 2---------------------------------")
    print("Transaction Reduction: ")
    t=time.time()
    rules=apriori_improvised_transactionreduction(transactions,min_sup,min_conf)
    e=time.time()
    total_time=e-t
    print(total_time)
    #for task 1.3
    print("Overall Time taken by different Approaches for ",filename)
    print("Overall time for inbuilt apriori:",overall_time_inbuilt)
    print("Overall time for apriori implementation from scratch: ",overall_time1)
    print("Total time taken by partitioning:",total_time_partition)
    print("Total time taken by Transaction Reduction:",total_time)



# In[43]:

"""
print("----------------------Apriori Algorithm -Task 1-----------------------------------------")
print("Dataset used : Sign ;Value of Minimum Support : 0.70 ;Value of Minimum Confidence: 0.50")
set_1("sign.txt",0.7,0.5)
print("----------------------Apriori Algorithm -Task 1-----------------------------------------")
print("Dataset used : FIFA;Value of Minimum Support : 0.70 ;Value of Minimum Confidence: 0.50")
set_1("fifa.txt",0.7,0.5)

print("----------------------Apriori Algorithm -Task 1-----------------------------------------")
print("Dataset used : Bible ;Value of Minimum Support : 0.70 ;Value of Minimum Confidence: 0.50")
set_1("bible.txt",0.7,0.5)

print("----------------------Apriori Algorithm -Task 1-----------------------------------------")
print("Dataset used : Kosarak ;Value of Minimum Support : 0.70 ;Value of Minimum Confidence: 0.50")
set_1("kosarak.txt",0.7,0.5)

print("----------------------Apriori Algorithm -Task 1-----------------------------------------")
print("Dataset used : Leviathian ;Value of Minimum Support : 0.70 ;Value of Minimum Confidence: 0.50")
set_1("leviathian.txt",0.7,0.5)

#for different value of support and confidence
print("Dataset used : Sign ;Value of Minimum Support : 0.60 ;Value of Minimum Confidence: 0.50")
set_1("sign.txt",0.6,0.5)

print("Dataset used : Sign ;Value of Minimum Support : 0.50 ;Value of Minimum Confidence: 0.50")
set_1("sign.txt",0.5,0.5)
"""
print("Dataset used : Sign ;Value of Minimum Support : 0.60 ;Value of Minimum Confidence: 0.50")
set_1("sign.txt",0.4,0.5)
"""
print("Dataset used : Sign ;Value of Minimum Support : 0.50 ;Value of Minimum Confidence: 0.50")
set_1("sign.txt",0.8,0.5)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
"""



