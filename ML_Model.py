#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Muhammad Awais
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Read the input CSV file
df = pd.read_csv('student-por.csv', header = None, names  = ['data'])
# Split the data in each row into separate columns based on a semicolon
df[['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','traveltime','studytime','failures','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic','famrel','freetime','goout','Dalc','Walc','health','absences','G1','G2','G3']] = df['data'].str.split(';', expand=True)

# Drop the original data column to add new updated columns
df.drop('data', axis=1, inplace=True)

# Split the dataset into a training set(70%) and a testing set(30%)
train_df, test_df = train_test_split(df, test_size=0.3)
train_df.to_csv('train_set_student.csv', index=False)

df = pd.read_csv('train_set_student.csv')
#header occurs 2 times in data so we remove one header
#find the index where the header occurs in the middle of the data
header_index = df[df['age'] == 'age'].index[0]

#slice the DataFrame excluding the row with the header
train_df = pd.concat([df.iloc[:header_index], df.iloc[header_index+1:]])

#Check for any outliers or inconsistencies in the data and remove them if necessary.
#train_df.info()
train_df.drop(["traveltime","failures","schoolsup","famsup","nursery","higher","internet","famrel","goout","Dalc","Walc"],axis=1,inplace=True)

#test_df.info()
test_df.drop(["traveltime","failures","schoolsup","famsup","nursery","higher","internet","famrel","goout","Dalc","Walc"],axis=1,inplace=True)

# Convert categorical variables into numerical variables in the training set
categorical_variables = {
    'school': {'MS': 0, 'GP': 1},
    'sex': {'"M"': 0, '"F"': 1},
    'address': {'"U"':0, '"R"': 1},
    'famsize' : {'"GT3"':0, '"LE3"': 1},
    'Pstatus' : {'"T"':0, '"A"': 1},
    'Mjob' : {'"at_home"':0, '"health"': 1, '"services"': 2, '"teacher"':3,'"other"':4},
    'Fjob' : {'"at_home"':0, '"health"': 1, '"services"': 2, '"teacher"':3,'"other"':4},
    'reason' : {'"course"':0, '"home"': 1, '"reputation"': 2, '"other"':3} ,
    'guardian': {'"mother"': 0, '"father"': 1, '"other"':2},
    'paid': {'"yes"': 0, '"no"': 1},
    'activities': {'"yes"': 0, '"no"': 1},
    'romantic': {'"yes"': 0, '"no"': 1}
}

for column, mapping in categorical_variables.items():
    train_df[column] = train_df[column].map(mapping)

for column, mapping in categorical_variables.items():
    test_df[column] = test_df[column].map(mapping)

#Opearions on train dataset    
train_df['G1'] = train_df['G1'].apply(lambda x: int(x.strip('"')))
train_df['G2'] = train_df['G2'].apply(lambda x: int(x.strip('"')))
train_df = train_df.astype(int)
#Opearions on test dataset 
#Change numbers from "1","2" string format to 1,2 string format
test_df['G1'] = test_df['G1'].apply(lambda x: int(x.strip('"')))
test_df['G2'] = test_df['G2'].apply(lambda x: int(x.strip('"')))

test_df = test_df.astype(int)
#normalization
#select columns whose values are large
normalization_cols = ['age', 'freetime', 'health', 'absences', 'G1', 'G2']

# Define the normalization function
def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# Apply the normalization function
train_df[normalization_cols] = train_df[normalization_cols].apply(normalize)
test_df[normalization_cols] = test_df[normalization_cols].apply(normalize)


grade_ranges = [
    (90, 100, 4),
    (80, 89, 3),
    (70, 79, 2),
    (60, 69, 1),
    (0, 59, 0)
]
train_df.info()

# Function to map marks to grades
def map_marks_to_grade(marks):
    for min_val, max_val, grade in grade_ranges:
        if min_val <= marks/20*100 <= max_val:
            return grade

train_df_Y = train_df['G3'].apply(map_marks_to_grade)
train_df_X = train_df.drop('G3', axis=1)

test_df_Y = test_df['G3'].apply(map_marks_to_grade)
test_df_X = test_df.drop('G3', axis=1)



# Multiple linear regression model
reg_model = LinearRegression()
reg_model.fit(train_df_X, train_df_Y)
y_pred = reg_model.predict(test_df_X)

# Evaluate Errors on Multiple linear regression model
mse = mean_squared_error(test_df_Y, y_pred)
mae = mean_absolute_error(test_df_Y, y_pred)
r2 = r2_score(test_df_Y, y_pred)
print("Mean squared error of Multiple linear regression model :", mse)
print("Mean absolute error of Multiple linear regression model :", mae)
print("R2 error of Multiple linear regression model :", r2, "\n")

#Calculate Accuracy, Precision, Recall and F1 score to assess the performance of the classifier
#Multiple linear regression model
mr_pred_grades =np.clip(np.round(y_pred), 0, 4).astype(int)
print("Accuracy, Precision, Recall and F1 score to assess the performance of the classifier on Multiple linear regression model")
# Accuracy
accuracy = accuracy_score(test_df_Y, mr_pred_grades)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(test_df_Y, mr_pred_grades, average='weighted',zero_division=1)
print("Precision:", precision)

# Recall
recall = recall_score(test_df_Y, mr_pred_grades, average='weighted',zero_division=1)
print("Recall:", recall)

# F1 Score
f1 = f1_score(test_df_Y, mr_pred_grades, average='weighted',zero_division=1)
print("F1 Score:", f1)


#Decision tree model
model = DecisionTreeRegressor(max_depth=3)
model.fit(train_df_X, train_df_Y)
predictions = model.predict(test_df_X)

# Evaluate Errors on Decision tree model
msed = mean_squared_error(test_df_Y, predictions)
maed = mean_absolute_error(test_df_Y, predictions)
r2d = r2_score(test_df_Y, predictions)
print('\n')
print("Mean squared error of Multiple Decision tree model :", msed)
print("Mean absolute error of Multiple Decision tree model :", maed)
print("R2 error of Multiple Decision tree model :", r2d , "\n")


dt_pred_grades =np.clip(np.round(predictions), 0, 4).astype(int)
print("Accuracy, Precision, Recall and F1 score to assess the performance of the classifier on decision tree model")
# Accuracy
accuracy = accuracy_score(test_df_Y, dt_pred_grades)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(test_df_Y, dt_pred_grades, average='weighted',zero_division=1)
print("Precision:", precision)

# Recall
recall = recall_score(test_df_Y, dt_pred_grades, average='weighted',zero_division=1)
print("Recall:", recall)

# F1 Score
f1 = f1_score(test_df_Y, dt_pred_grades, average='weighted',zero_division=1)
print("F1 Score:", f1)




def predict_grades(input_data):
    rf_pred = model.predict(input_data)
    print("Predicted grades are:   ",rf_pred)
    return rf_pred

import tkinter as tk

window = tk.Tk()
window.geometry("800x800")
window.configure(background='#303030')
window.title("Student Information Form")

def store_input():
    inputs = [] 
    inputs.append(int(school_entry.get()))
    inputs.append(int(sex_entry.get()))
    inputs.append(int(age_entry.get()))
    inputs.append(int(address_entry.get()))
    inputs.append(int(famsize_entry.get()))
    inputs.append(int(pstatus_entry.get()))
    inputs.append(int(medu_entry.get()))
    inputs.append(int(fedu_entry.get()))
    inputs.append(int(mjob_entry.get()))
    inputs.append(int(fjob_entry.get()))
    inputs.append(int(resason_entry.get()))
    inputs.append(int(guardian_entry.get()))
    inputs.append(int(studytime_entry.get()))
    inputs.append(int(paid_entry.get()))
    inputs.append(int(activities_entry.get()))
    inputs.append(int(romantic_entry.get()))
    inputs.append(int(freetime_entry.get()))
    inputs.append(int(health_entry.get()))
    inputs.append(int(absences_entry.get()))
    inputs.append(int(G1_entry.get()))
    inputs.append(int(G2_entry.get()))
    col_names = ['school','sex','age','address','famsize','Pstatus','Medu','Fedu','Mjob','Fjob','reason','guardian','studytime'
                ,'paid','activities','romantic','freetime','health','absences','G1','G2']
    df=pd.DataFrame()
    
    for col, val in zip(col_names, inputs):
        df[col] = [val]
    df.info()
    mystr=""
    ans1=predict_grades(df)
    ans=round(float(ans1[0]))
    if ans == 0:
        mystr="D"
    elif ans == 1:
        mystr="C"
    elif ans == 2:
        mystr="B"
    elif ans == 3:
        mystr="A"
    elif ans == 4 or ans > 4:
        mystr="A+"
    result_label.configure(text=f"Predicited grade of G3: {mystr}!")

    
custom_font = ("Helvetica", 16)
    
# school input field
school_label = tk.Label(window, text="School:",font=custom_font, fg="#A0D6B4",bg="#303030")
school_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
school_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
school_entry.grid(row=0, column=1, padx=10, pady=10)

# sex input field
sex_label = tk.Label(window, text="Sex:" ,font=custom_font, fg="#A0D6B4",bg="#303030")
sex_label.grid(row=0, column=2, padx=10, pady=10, sticky=tk.W)
sex_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
sex_entry.grid(row=0, column=3, padx=10, pady=10)

# age input field
age_label = tk.Label(window, text="Age:",font=custom_font, fg="#A0D6B4",bg="#303030")
age_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
age_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
age_entry.grid(row=1, column=1, padx=10, pady=10)

# address input field
address_label = tk.Label(window, text="Address:",font=custom_font, fg="#A0D6B4",bg="#303030")
address_label.grid(row=1, column=2, padx=10, pady=10, sticky=tk.W)
address_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
address_entry.grid(row=1, column=3, padx=10, pady=10)

# famsize input field
famsize_label = tk.Label(window, text="Family Size:",font=custom_font, fg="#A0D6B4",bg="#303030")
famsize_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
famsize_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
famsize_entry.grid(row=2, column=1, padx=10, pady=10)

# Pstatus input field
pstatus_label = tk.Label(window, text="Parent's Cohabitation Status:",font=custom_font, fg="#A0D6B4",bg="#303030")
pstatus_label.grid(row=2, column=2, padx=10, pady=10, sticky=tk.W)
pstatus_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
pstatus_entry.grid(row=2, column=3, padx=10, pady=10)

# Medu input field
medu_label = tk.Label(window, text="Mother's Education:",font=custom_font, fg="#A0D6B4",bg="#303030")
medu_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
medu_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
medu_entry.grid(row=3, column=1, padx=10, pady=10)

# Fedu input field
fedu_label = tk.Label(window, text="Father's Education:",font=custom_font, fg="#A0D6B4",bg="#303030")
fedu_label.grid(row=3, column=2, padx=10, pady=10, sticky=tk.W)
fedu_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
fedu_entry.grid(row=3, column=3,padx=10, pady=10)
#Mother Job field
mjob_label = tk.Label(window, text="Mother's Job:",font=custom_font, fg="#A0D6B4",bg="#303030")
mjob_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
mjob_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
mjob_entry.grid(row=4, column=1, padx=10, pady=10)
#Father Job field
fjob_label = tk.Label(window, text="Father's Job:",font=custom_font, fg="#A0D6B4",bg="#303030")
fjob_label.grid(row=4, column=2, padx=10, pady=10, sticky=tk.W)
fjob_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
fjob_entry.grid(row=4, column=3, padx=10, pady=10)
#Reason Field
resason_label = tk.Label(window, text="Reason:",font=custom_font, fg="#A0D6B4",bg="#303030")
resason_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
resason_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
resason_entry.grid(row=5, column=1, padx=10, pady=10)
#guardian field
gurdian_label = tk.Label(window, text="Guardian:",font=custom_font,fg="#A0D6B4",bg="#303030")
gurdian_label.grid(row=5, column=2, padx=10, pady=10, sticky=tk.W)
guardian_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
guardian_entry.grid(row=5, column=3, padx=10, pady=10)

#study time field
studytime_label = tk.Label(window, text="Study Time:",font=custom_font, fg="#A0D6B4",bg="#303030")
studytime_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)
studytime_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
studytime_entry.grid(row=6, column=1, padx=10, pady=10)

#Paid field
paid_label = tk.Label(window, text="Paid:",font=custom_font, fg="#A0D6B4",bg="#303030")
paid_label.grid(row=6, column=2, padx=10, pady=10, sticky=tk.W)
paid_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
paid_entry.grid(row=6, column=3, padx=10, pady=10)

#activities
activities_label = tk.Label(window, text="Activities:",font=custom_font, fg="#A0D6B4",bg="#303030")
activities_label.grid(row=7, column=0, padx=10, pady=10, sticky=tk.W)
activities_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
activities_entry.grid(row=7, column=1, padx=10, pady=10)

#romantic field
romantic_label = tk.Label(window, text="Romantic:",font=custom_font, fg="#A0D6B4",bg="#303030")
romantic_label.grid(row=7, column=2, padx=10, pady=10, sticky=tk.W)
romantic_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
romantic_entry.grid(row=7, column=3, padx=10, pady=10)

#freetime field
freetime_label = tk.Label(window, text="Free Time:",font=custom_font, fg="#A0D6B4",bg="#303030")
freetime_label.grid(row=8, column=0, padx=10, pady=10, sticky=tk.W)
freetime_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
freetime_entry.grid(row=8, column=1, padx=10, pady=10)


#health field
health_label = tk.Label(window, text="Health:",font=custom_font, fg="#A0D6B4",bg="#303030")
health_label.grid(row=8, column=2, padx=10, pady=10, sticky=tk.W)
health_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
health_entry.grid(row=8, column=3, padx=10, pady=10)


#absences field
absences_label = tk.Label(window, text="Absences:",font=custom_font, fg="#A0D6B4",bg="#303030")
absences_label.grid(row=9, column=0, padx=10, pady=10, sticky=tk.W)
absences_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
absences_entry.grid(row=9, column=1, padx=10, pady=10)


#G1 field
G1_label = tk.Label(window, text="G1:",font=custom_font, fg="#A0D6B4",bg="#303030")
G1_label.grid(row=9, column=2, padx=10, pady=10, sticky=tk.W)
G1_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
G1_entry.grid(row=9, column=3, padx=10, pady=10)

#G2 field
G2_label = tk.Label(window, text="G2:",font=custom_font, fg="#A0D6B4",bg="#303030")
G2_label.grid(row=10, column=0, padx=10, pady=10, sticky=tk.W)
G2_entry = tk.Entry(window,foreground='#728FCE', background='white', borderwidth=3, relief='groove', font=('Arial', 12))
G2_entry.grid(row=10, column=1, padx=10, pady=10)


result_label = tk.Label(window, text="")
result_label.grid(row=11, column=1, columnspan=2, padx=10, pady=10)


submit_button = tk.Button(window, text="Submit", command=store_input, bg="#4CAF50", fg="Black", font=("Arial", 14), borderwidth=2)
submit_button.grid(row=12, column=1, columnspan=2, padx=10, pady=10)


window.mainloop()




