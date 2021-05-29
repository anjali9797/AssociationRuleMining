#!/usr/bin/env python
# coding: utf-8

# In[11]:


#importing packages
import pandas as pd
import itertools 
import numpy as np
import time
import logging
import sys
import csv
from collections import defaultdict
from itertools import chain, combinations
import pyfpgrowth
import operator


# In[12]:


#designing a class for fp tree node
class fp_node:
    
    def __init__(self, item , item_count=0, parent=None, link=None):
        #initialising the various items of the root node
        #will be useful in optimisation approach
        self.visited=0#will be useful for merging approach
        #common attributes
        self.parent=parent
        self.link=link
        self.item=item
        self.item_count=item_count
        self.child={}


# In[13]:


#the normal fpgrowth algorithm
class fp_tree:
    def __init__(self , name ,min_support ,min_conf):
        #storing transactions
        self.data = name
        self.root = fp_node('Null' ,0)
        self.node_table = list()
        self.wordlinesort= list()
        self.full_freq = defaultdict(int)
        self.freq = defaultdict(int)
        self.item_order = defaultdict(int)
        self.min_support = int(min_support*len(self.data)) 
        self.min_confidence = min_conf
        self.min_sup=min_support
        self.lendata=len(name)
        self.build_tree(self.data)
        print("FP Tree Built")
        print("Finding Frequent Patterns")
        self.pat = self.create_freq_patterns()
        print("Completed")
    

    
    
    def insert_transection(self,transection):
        #inserting a new transaction which is valid
        lentran=len(transection)
        num_items=0
        sorted_minsup_items = sorted(transection ,key=lambda i: self.item_order[i])
        root = self.root
        self.wordlinesort.append(sorted_minsup_items)
        num_items=lentran
        #print(lentran)#length of inp
        new_inser=0
        for i in sorted_minsup_items:
            if i not in root.child.keys():
                #creating a node
                root.child[i] = fp_node(i ,1 ,root ,None)
                new_inser+=1
                root = root.child[i]
                for item_detail in self.node_table:
                    if item_detail['item'] != root.item:
                        continue
                    else:
                        if item_detail['linkage'] is not None:
                            l = item_detail['linkage']
                            while( l.link is not None):
                                l = l.link
                            l.link = root
                        else:
                             item_detail['linkage'] = root

            else:
                root.child[i].item_count += 1
                root = root.child[i]

        
    def build_tree(self ,data):
        for i in transactions:
            for j in i:
            #increment full_freq
                self.full_freq[j] = self.full_freq[j] + 1
        freq_set = defaultdict(int)   
    #check 1 done
        for key in self.full_freq.keys():
             if self.full_freq[key] >= int(self.min_sup*self.lendata):
                freq_set[key] = self.full_freq[key]
    #sorting them out for the ll1 table  
    #print("freq_set",freq_set) #check2
        counter = 0
        self.freq=dict(sorted(freq_set.items(), key=operator.itemgetter(1),reverse=True))
    #creating the linkage table
        for key in self.freq.keys():
            item_id=key
            #print(item_id)
            count_item=self.freq[key]
            #print(freq[key])
            self.item_order[key] = counter 
            counter=counter+1
            self.node_table.append({'item':item_id ,'item_freq':count_item ,'linkage':None}) 
        
        ## Construct FPTree 
        count_line=0
        for line in data:
            count_line=count_line+1
            min_sup_item = list()
            for item in line:
                if item not in self.freq.keys():
                    continue
                else:
                    min_sup_item.append(item)
            
            #insert transection to the fp tree
            if len(min_sup_item) >= 1 :
                self.insert_transection(min_sup_item)
        #print("Count:",count_line)
    

            
    def create_conditional_pattern_base(self):
        #reversing the table
        pattern_base = defaultdict(int)
        pb = defaultdict(int)
        rev_table = self.node_table[::-1]
        for itm in rev_table:
            paths = list()
            node = itm['linkage']
            while node is not None:
                prev = node.parent
                trans = list()
                while prev.parent is not None:
                    trans.append(prev)
                    prev = prev.parent
                lentrans=len(trans)
                if(lentrans>=1):
                    if node in pattern_base.keys():
                        pattern_base[ node ].append([ set(trans),node.item_count] )
                    else:
                         pattern_base[ node ] = [ [ set(trans),node.item_count] ]
                node = node.link

        for i in pattern_base.keys():
            l = pattern_base[i]
            for j in l:
                temp=list()
                it = j[0]
                for k in it:
                    temp.append(k.item)
            tset=set(temp)
            count=j[1]
            if i.item  in pb.keys():
                pb[i.item].append( [tset,count] )
            else:
                pb[i.item] = [ [tset,count] ]

        return pb
    

    
    def create_freq_patterns(self):
        res=list()
        #main pbase
        cond_fptree = defaultdict(int) 
        pattern_base=self.create_conditional_pattern_base()
        for i in pattern_base.keys():
            l=pattern_base[i]
            freq=defaultdict(int)
            final=defaultdict(int)
            for j in l:
                it=j[0]
               #print(j)
                for k in it:
                    freq[k]+=j[1]
            for j in freq.keys():
                if freq[j] < self.min_support:
                    continue
                else:
                    final[j]=freq[j]
            cond_fptree[i]=final
        
        for key in cond_fptree.keys():
            vals= cond_fptree[key]
            fqt_patterns=list()
            sets=list()
            val_keys=vals.keys()
            values=set(val_keys)
            val_len=len(values)
            for i in range(val_len):
                j=i+1
                sets+=list(itertools.combinations(values,j))
            for i in sets:
                list_val=list()
                for j in i:
                    temp=vals[j]
                    list_val.append(temp)
                min_val=min(list_val)
                pattern=set(i).union({key})
                fqt_patterns.append([pattern,min_val])
            res.append(fqt_patterns)
        return res


# In[14]:


#improvised fp growth algorithm
class fp_tree_merging_optimized:
    def __init__(self , name ,min_support ,min_conf):
        
        #Root 
        self.root = fp_node('Null' ,0)
        self.node_table = list()
        self.wordlinesort= list()
        self.full_freq = defaultdict(int)
        self.freq = defaultdict(int)
        self.item_order = defaultdict(int)
        self.data = name
        self.lendata=len(name)
        self.min_sup=min_support
        self.min_support = int(self.min_sup*self.lendata) 
        self.build_tree(self.data)
        print("FP Tree built using merging algorithm")
        print("FP pattern finding (improvised approach)")
        self.pattern = self.create_freq_patterns()
        print("Completed!")
        
        

    
    def insert_transection(self,transection):
        #inserting a new transaction which is valid
        lentran=len(transection)
        num_items=0
        sorted_minsup_items = sorted(transection ,key=lambda i: self.item_order[i])
        root = self.root
        self.wordlinesort.append(sorted_minsup_items)
        num_items=lentran
        #print(lentran)#length of inp
        new_inser=0
        for i in sorted_minsup_items:
            if i not in root.child.keys():
                #creating a node
                root.child[i] = fp_node(i ,1 ,root ,None)
                new_inser+=1
                root = root.child[i]
                for item_detail in self.node_table:
                    if item_detail['item'] != root.item:
                        continue
                    else:
                        if item_detail['linkage'] is not None:
                            l = item_detail['linkage']
                            while( l.link is not None):
                                l = l.link
                            l.link = root
                        else:
                             item_detail['linkage'] = root

            else:
                root.child[i].item_count += 1
                root = root.child[i]

   
    def build_tree(self ,data):
        for i in transactions:
            for j in i:
            #increment full_freq
                self.full_freq[j] = self.full_freq[j] + 1
        freq_set = defaultdict(int)   
    #check 1 done
        for key in self.full_freq.keys():
             if self.full_freq[key] >= int(self.min_sup*self.lendata):
                    freq_set[key] = self.full_freq[key]
    #sorting them out for the ll1 table  
    #print("freq_set",freq_set) #check2
        counter = 0
        self.freq=dict(sorted(freq_set.items(), key=operator.itemgetter(1),reverse=True))
    #creating the linkage table
        for key in self.freq.keys():
            item_id=key
            #print(item_id)
            count_item=self.freq[key]
            #print(freq[key])
            self.item_order[key] = counter 
            counter=counter+1
            self.node_table.append({'item':item_id ,'item_freq':count_item ,'linkage':None}) 
        
        ## Construct FPTree 
        count_line=0
        for line in data:
            count_line=count_line+1
            min_sup_item = list()
            for item in line:
                if item not in self.freq.keys():
                    continue
                else:
                    min_sup_item.append(item)
            
            #insert transection to the fp tree
            if len(min_sup_item) >= 1 :
                self.insert_transection(min_sup_item)
        #print("Count:",count_line)


    def create_conditional_pattern_base(self):
        #reversing the table
        pattern_base = defaultdict(int)
        pb = defaultdict(int)
        rev_table = self.node_table[::-1]
        for itm in rev_table:
            paths = list()
            node = itm['linkage']
            while node is not None:
                prev = node.parent
                trans = list()
                while prev.parent is not None:
                    trans.append(prev)
                    prev = prev.parent
                lentrans=len(trans)
                if(lentrans>=1 and node.visited != 1):
                    if node in pattern_base.keys():
                        pattern_base[ node ].append([ set(trans),cur_node.item_count] )
                    else:
                         pattern_base[ node ] = [ [ set(trans),node.item_count] ]
                    counter=0
                    node.visited=1
                    for i in trans:
                        remains=trans[counter+1: ]
                        lenremains=len(remains)
                        if(lenremains>=1):
                            if i.visited!=1:
                                    if trans[counter] in pattern_base.keys():
                                        pattern_base[trans[counter]].append([set(remains),i.item_count])
                                    else:
                                        pattern_base[trans[counter]]=[[set(remains),i.item_count]]
                            i.visited=1
                        counter +=1

                node = node.link

        for i in pattern_base.keys():
            l = pattern_base[i]
            for j in l:
                temp=list()
                it = j[0]
                for k in it:
                    temp.append(k.item)
            tset=set(temp)
            count=j[1]
            if i.item  in pb.keys():
                pb[i.item].append( [tset,count] )
            else:
                pb[i.item] = [ [tset,count] ]

        return pb
    

    
    def create_freq_patterns(self):
        res=list()
        #main pbase
        cond_fptree = defaultdict(int) 
        pattern_base=self.create_conditional_pattern_base()
        for i in pattern_base.keys():
            l=pattern_base[i]
            freq=defaultdict(int)
            final=defaultdict(int)
            for j in l:
                it=j[0]
               #print(j)
                for k in it:
                    freq[k]+=j[1]
            for j in freq.keys():
                if freq[j] < self.min_support:
                    continue
                else:
                    final[j]=freq[j]
            cond_fptree[i]=final
        
        for key in cond_fptree.keys():
            vals= cond_fptree[key]
            fqt_patterns=list()
            sets=list()
            val_keys=vals.keys()
            values=set(val_keys)
            val_len=len(values)
            for i in range(val_len):
                j=i+1
                sets+=list(itertools.combinations(values,j))
            for i in sets:
                list_val=list()
                for j in i:
                    temp=vals[j]
                    list_val.append(temp)
                min_val=min(list_val)
                pattern=set(i).union({key})
                fqt_patterns.append([pattern,min_val])
            res.append(fqt_patterns)
        #print(res)
        return res


# In[16]:


#running it on sign.txt data
print("Running on Sign Dataset")
file = open("sign.txt", 'r') 
data = file.readlines() 
print(len(data))
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
print(len(transactions))
print("File Read Successfully ! Data Stored in list : transactions")
print("Using Inbuilt FPGrowth Algorithm")
start = time.time()
threshold=0.5*730
#we can input the values of support and confidence but here i am taking a support of 70% and confidence of 50%
patterns = pyfpgrowth.find_frequent_patterns(transactions,threshold)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
end = time.time()
overall_time1=end-start
print("Overall time taken by inbuilt :",overall_time1)
s=time.time()
fp_tree(transactions ,0.5 ,0.7)
e = time.time()
overall_time2=e-s
print("Overall time taken by fptree from scratch: ",overall_time2)
s=time.time()
fp_tree_merging_optimized(transactions ,0.5,0.7)
e = time.time()
overall_time3=e-s
print("Overall time taken by optimized fptree: ",overall_time3)


# In[17]:


#running on kosarak dataset
print("Running on Kosarak dataset")
file = open("kosarak.txt", 'r') 
data = file.readlines() 
print(len(data))
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
print(len(transactions))
print("File Read Successfully ! Data Stored in list : transactions")
print("Using Inbuilt FPGrowth Algorithm")
start = time.time()
threshold=0.5*730
#we can input the values of support and confidence but here i am taking a support of 70% and confidence of 50%
patterns = pyfpgrowth.find_frequent_patterns(transactions,threshold)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
end = time.time()
overall_time1=end-start
s=time.time()
fp_tree(transactions ,0.5 ,0.7)
e = time.time()
overall_time2=e-s
print("Overall time: ",overall_time2)
s=time.time()
fp_tree_merging_optimized(transactions ,0.5,0.7)
e = time.time()
overall_time3=e-s
print("Overall time: ",overall_time3)


# In[18]:


#running on fifa dataset
print("Running on FIFA dataset")
file = open("fifa.txt", 'r') 
data = file.readlines() 
print(len(data))
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
print(len(transactions))
print("File Read Successfully ! Data Stored in list : transactions")
print("Using Inbuilt FPGrowth Algorithm")
start = time.time()
threshold=0.5*730
#we can input the values of support and confidence but here i am taking a support of 70% and confidence of 50%
#patterns = pyfpgrowth.find_frequent_patterns(transactions,threshold)
#rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
end = time.time()
overall_time1=end-start
s=time.time()
fp_tree(transactions ,0.5 ,0.7)
e = time.time()
overall_time2=e-s
print("Overall time: ",overall_time2)
s=time.time()
fp_tree_merging_optimized(transactions ,0.5,0.7)
e = time.time()
overall_time3=e-s
print("Overall time: ",overall_time3)


# In[ ]:
print("Running on Bible dataset")
file = open("bible.txt", 'r') 
data = file.readlines() 
print(len(data))
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
print(len(transactions))
print("File Read Successfully ! Data Stored in list : transactions")
s=time.time()
fp_tree(transactions ,0.5 ,0.7)
e = time.time()
overall_time2=e-s
print("Overall time: ",overall_time2)
s=time.time()
fp_tree_merging_optimized(transactions ,0.5,0.7)
e = time.time()
overall_time3=e-s
print("Overall time: ",overall_time3)




# In[ ]:
print("Running on Leviathian dataset")
file = open("leviathian.txt", 'r') 
data = file.readlines() 
print(len(data))
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
print(len(transactions))
print("File Read Successfully ! Data Stored in list : transactions")
s=time.time()
fp_tree(transactions ,0.5 ,0.7)
e = time.time()
overall_time2=e-s
print("Overall time: ",overall_time2)
s=time.time()
fp_tree_merging_optimized(transactions ,0.5,0.7)
e = time.time()
overall_time3=e-s
print("Overall time: ",overall_time3)







# In[ ]:





# In[ ]:





# In[ ]:




