#!/usr/bin/env python
# coding: utf-8

# # Αποθήκες και εξόρυξη δεδομένων
# 
#  Το έργο του στοχεύει στη χρήση σύγχρονων και αποτελεσματικών τεχνικών όπως τον αλγόριθμο KNN που ομαδοποιεί το σύνολο δεδομένων και παρέχει την ολοκληρωμένη και γενική προσέγγιση για τη σύσταση κρασιού στους πελάτες βάσει ορισμένων χαρακτηριστικών. Αυτό θα βοηθήσει τους ιδιοκτήτες των καταστημάτων να γνωρίζουν ήδη για τη ζήτηση του κρασιού και κατά συνέπεια το απόθεμα θα ενημερωθεί. Αυτό θα διευκολύνει την αύξηση του κέρδους των ιδιοκτητών καταστημάτων. 
# 
# 

# In[104]:


import pickle
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans


# In[105]:


data=pd.read_csv('/Users/paolomarco/Downloads/wine_data.csv');#Κανουμε import το dataset


# In[106]:


data.describe()# μας δίνει κάποια γενικά στατιστικά του dataset


# In[107]:


data.head()# εμφανίζει τα πρώτα rows του dataset 


# # Preprocessing των δεδομένων 
# * normalization
# * Split τα δεδομένα 
# * scalling

# In[108]:


from sklearn.model_selection import train_test_split


# In[109]:


X = data.iloc[:, 1:13].values
y = data.iloc[:, 0].values
print(X)
print(y)
X= (X - np.min(X)) / (np.max(X) - np.min(X))
print(X)


# In[110]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)


# In[111]:


from sklearn.preprocessing import StandardScaler #κάνουμε scale τα δεδομένα 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # ΚΝΝ algorithm (nearest-neighbor)
# 

# In[112]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
knn = []
for i in range(1,21):
            
    classifier = KNeighborsClassifier(n_neighbors=i)
    trained_model=classifier.fit(X_train,y_train)
    trained_model.fit(X_train,y_train )
    
    # Πρόβλεψη  των Test  results
    
    y_pred = classifier.predict(X_test)
    
    # Δημιουργία Confusion Matrix
    
    from sklearn.metrics import confusion_matrix
    
    cmat_KNN = confusion_matrix(y_test, y_pred)
    print(i)
    print(cmat_KNN)
    print("Accuracy train score for algorithm  KNN",accuracy_score(y_train, trained_model.predict(X_train))*100)
    
    
    print("Accuracy test score for algorithm  KNN ",accuracy_score(y_test, y_pred)*100)
   
    knn.append(accuracy_score(y_test, y_pred)*100)
    print(classification_report(y_test, y_pred))
    
  


# In[113]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))  
plt.plot(range(1, 21),knn, color='cyan', linestyle='dashed', marker='o',  
             markerfacecolor='blue', markersize=15)
plt.title('Accuracy για διαφορετικές τιμές του Κ')  
plt.xlabel('τιμές K')  
plt.ylabel('Accuracy') 


# # SUPORT VECTOR MACHINE ALGORITHM

# In[114]:


# εφαρμογη SVM στο Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
trained_model=classifier.fit(X_train,y_train)
trained_model.fit(X_train,y_train )


# Πρόβλεψη  των Test  results
y_pred = classifier.predict(X_test)

#  Δημιουργεία Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_SVM = confusion_matrix(y_test, y_pred)
print(cm_SVM)
print("Accuracy train train score for  SVM algorithm",accuracy_score(y_train, trained_model.predict(X_train))*100)

print("Accuracy test score for  SVM algorithm",accuracy_score(y_test, y_pred)*100)


# # Random Forest

# In[115]:


from sklearn.ensemble import RandomForestRegressor
# Δινουμε ως ορισμα  1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train το μοντέλο 
rf.fit(X_train, y_train);


# In[116]:


# Πρόβλεψη  των Test  results
predictions = rf.predict(X_test)
# Calculate the absolute errors
errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))


# In[117]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Υπολογισμός accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 20), '%.')


# In[ ]:




