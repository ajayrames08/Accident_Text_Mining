"""
__author__ =
Ajaykumar Rameshwaram - A0134596W
Akbar Ahmed - A0134484A
Divya Jawahar – A0134518H
Kadirvel Mani Chandrika - A0134564E
Sangeetha Sambamoorthy - A0134591E
Santhosh Kumar Ponnambalam -A0134503U
"""
import necessary
import csv
import re
from sklearn.metrics import classification_report
from operator import itemgetter
'''
Method Name: dictionary
Returns: dictionary
@Params: text of similar cases, Total Cases, List of Words to keep in dictionary
Functionality:Creates an individual dictionary for all 11 different cases for the classification
of files
'''

def dictionary(text,total_cases,verbs):
    text_nopunc=text.translate(necessary.string.maketrans("",""), necessary.string.punctuation)
    text_lower=text_nopunc.lower()
    stop = necessary.stopwords.words('english')
    stpwords_extra=["victim","co-worker","worker","employee","employees","die","dead","death","accident","injured","died"];
    for i in stpwords_extra:
        stop.append(i)
    text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
    tokens = necessary.word_tokenize(text_nostop)
    wnl = necessary.nltk.WordNetLemmatizer()
    text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
    tokens_lem = necessary.word_tokenize(text_lem)
    verbs=" ".join([wnl.lemmatize(t) for t in verbs])
    tokens_lem=[x for x in tokens_lem if x in verbs]
    counts = necessary.collections.Counter(tokens_lem)
    final_words_count=[]
    for i in tokens_lem:
        final_words_count.append([i,float(counts[i])/float(total_cases)])
    best = sorted(final_words_count, key=itemgetter(1), reverse=True)[20:]
    dictionary={k:v for k,v in best}
    return dictionary

'''
Method Name: keepverbs
Returns: verbs and nouns in a text
@Params: text
Functionality:Return Nouns and verbs which makes more meaning
'''

def keepverbs(text):
    text = necessary.word_tokenize(text)
    text_pos=necessary.nltk.pos_tag(text)
    verbs_nouns=[wt[0] for wt in text_pos if (wt[1] == 'VBG' or wt[1]=='VBD' or wt[1]=='VG' or wt[1]=='VD' or wt[1]=="NN" or wt[1]=="NNS")]
    return verbs_nouns
'''
Method Name: data_preperation
Returns: addition of text with similar causes
@Params: cause,data
Functionality: Form a text with collection of all similar causes
'''


def data_preperation(cause,data):
    Caught_unclean_cause=data[data.Cause.isin(cause)]
    Caught_unclean_summary_case=""
    total_cases=0
    for i in Caught_unclean_cause.index:
        Caught_unclean_summary_case=Caught_unclean_summary_case+str(Caught_unclean_cause.Summary[i])
        total_cases=total_cases+1
    return Caught_unclean_summary_case,total_cases
'''
Method Name: dictionary_creation
Returns: dictionary
@Params: cause,data
Functionality: Form a dictonary with collection of all similar causes with normalized values
'''

def dictionary_creation(summary,train_data):
    cause1,no_cases=data_preperation(summary,train_data)
    verbs=keepverbs(cause1)
    dict_final=dictionary(cause1,no_cases,verbs)
    return dict_final
'''
Method Name: process_test_data
Returns: predictions
@Params: test data and dictionaries
Functionality: classifies test data to the cases
'''

def process_test_data(test_data,dict_caught_btw_objects,dict_collapse_of_object,dict_drowning,dict_electrocution,dict_chemical_substances,dict_extereme_temperature,dict_falls,dict_fires,dict_others,dict_struck,dict_suffocation):
    predictions=[];
    stop=[];
    causes=["Caught in/between Objects","Collapse of object","Drowning","Electrocution","Exposure to Chemical Substances","Exposure to extreme temperatures","Falls","Fires and Explosion","Other","Struck By Moving Objects","Suffocation"]
    stpwords_extra=["victim","co-worker","worker","employee","employees","die","dead","death","accident","injured"]
    stop = necessary.stopwords.words('english')
    for i in stpwords_extra:
        stop.append(i)
    for index,row in test_data.iterrows():
        sample_text=row[2];
        text_nopunc=sample_text.translate(necessary.string.maketrans("",""), necessary.string.punctuation)
        text_lower=text_nopunc.lower()
        sumcases=[0,0,0,0,0,0,0,0,0,0,0];
        text_nostop=" ".join(filter(lambda word: word not in stop, text_lower.split()))
        tokens = necessary.word_tokenize(text_nostop)
        wnl = necessary.nltk.WordNetLemmatizer()
        text_lem=" ".join([wnl.lemmatize(t) for t in tokens])
        word_count_sentence=len(text_lem)
        tokens_lem = necessary.word_tokenize(text_lem)
        for i in tokens_lem:
            try:
                sumcases[0]+=dict_caught_btw_objects[i]
            except KeyError:
                pass
            try:
                sumcases[1]+=dict_collapse_of_object[i]
            except KeyError:
                pass
            try:
                sumcases[2]+=dict_drowning[i]
            except KeyError:
                pass
            try:
                sumcases[3]+=dict_electrocution[i]
            except KeyError:
                pass
            try:
                sumcases[4]+=dict_chemical_substances[i]
            except KeyError:
                pass
            try:
                sumcases[5]+=dict_extereme_temperature[i]
            except KeyError:
                pass
            try:
                sumcases[6]+=dict_falls[i]
            except KeyError:
                pass
            try:
                sumcases[7]+=dict_fires[i]
            except KeyError:
                pass
            try:
                sumcases[8]+=dict_others[i]
            except KeyError:
                pass
            try:
                sumcases[9]+=dict_struck[i]
            except KeyError:
                pass
            try:
                sumcases[10]+=dict_suffocation[i]
            except KeyError:
                pass
        predictions.append(causes[sumcases.index(max(sumcases))])
    return predictions
'''
Method Name: prior_occupations
Returns: occupation_final_list
@Params: test data
Functionality: Return the occupation from the summary
'''

def prior_occupation(test_data):
    occupation_final_list=[]
    for index,row in test_data.iterrows():
        text=row[2];
        text1=necessary.word_tokenize(text)
        train_data = necessary.pd.read_csv('/Users/AJ/PycharmProjects/Assignment_Text_Mining/OCCUPATIONS.csv')
        ''' Renaming the coloumns'''
        col_name_1 =train_data.columns[0]
        train_data=train_data.rename(columns = {col_name_1:'Occupation'})
        occupation=train_data.Occupation
        occupation_list=[]
        for i in occupation:
            occupation_list.append(i)
        occupation_list=[x.lower() for x in occupation_list]
        occupation_final=[x for x in text1 if x in occupation_list]
        occupation_final_list.append(occupation_final)
    return occupation_final_list
'''
Method Name: activities
Returns: activities_final_list
@Params: test data
Functionality: Return the activities from the summary
'''

def activities(test_data):
    activities_final_list=[]
    answer_tools=[]
    tools_final_list=[]
    for index,row in test_data.iterrows():
        text=row[2]
        text.replace("were","was",5)
        reasonable_text=re.findall(r'(?<=was).*?(?=\.){1}',text)
        reasonable_text=reasonable_text[:1]
        for text in reasonable_text:
            text1=necessary.word_tokenize(text)
            tokens=[]
            text_pos=necessary.nltk.pos_tag(text1)
            answer=[]
            pattern = "ACT: {<VBG><IN>?<TO>?<VB>?<RP>?<JJ>?<DT>?(<NN>|<NNS>)+}"
            NPChunker = necessary.nltk.RegexpParser(pattern)
            result = NPChunker . parse(text_pos)
            for subtree in result.subtrees(filter=lambda t: t.node == 'ACT'):
            # print the noun phrase as a list of part-of-speech tagged words
                for i in subtree.leaves():
                    answer.append(i[0])
                    result_last = ' '.join(answer)
                activities_final_list.append(result_last)
    return activities_final_list
'''
Method Name: tools
Returns: tools_final_list
@Params: test data
Functionality: Return the tools from the summary
'''

def tools(test_data):
    tools_final_list=[]
    for index,row in test_data.iterrows():
        text=row[1]
        text=text.lower()
        text=text.replace("from","by",1)
        text=text.replace("with","by",1)
        answer=[]
        reasonable_text=re.findall(r'(?<=by).*',text)
        reasonable_text=''.join(reasonable_text)
        text1=necessary.word_tokenize(reasonable_text)
        text_pos=necessary.nltk.pos_tag(text1)
        pattern = "OBJ: {<NN>*}"
        PChunker = necessary.nltk.RegexpParser(pattern)
        result = PChunker . parse(text_pos)
        for subtree in result.subtrees(filter=lambda t: t.node == 'OBJ'):
            # print the noun phrase as a list of part-of-speech tagged words
            for i in subtree.leaves():
                answer.append(i[0])
                result_last = ' '.join(answer)
            tools_final_list.append(result_last)
    return tools_final_list
'''
-------------------------------------------------------------------------------------------------------------------------------------
Main method
-------------------------------------------------------------------------------------------------------------------------------------
'''

if __name__ == '__main__':
    train_data = necessary.pd.read_csv('/Users/AJ/PycharmProjects/Assignment_Text_Mining/TrainMasia.csv')
    #Renaming the coloumns
    col_name_1 =train_data.columns[0]
    train_data=train_data.rename(columns = {col_name_1:'Cause'})
    col_name_2=train_data.columns[2]
    train_data=train_data.rename(columns = {col_name_2:'Summary'})
    #Read the Training Data
    #Dictionary for Caught in between objects
    summary=['Caught in/between Objects', 'caught objects']
    dict_caught_btw_objects=dictionary_creation(summary,train_data)
    #Dictionary for Collapse of objects
    summary=["Collapse of object"]
    dict_collapse_of_object=dictionary_creation(summary,train_data)
    #Dictionary for Drowning
    summary=["Drowning"]
    dict_drowning=dictionary_creation(summary,train_data)
    #Dictionary for Electrocution
    summary=["Electrocution"]
    dict_electrocution=dictionary_creation(summary,train_data)
    #Dictionary for Exposure to Chemical Substances
    summary=["Exposure to Chemical Substances"]
    dict_chemical_substances=dictionary_creation(summary,train_data)
    #Dictionary for Exposure to extreme temperatures
    summary=["Exposure to extreme temperatures"]
    dict_extereme_temperature=dictionary_creation(summary,train_data)
    #Dictionary for Falls
    summary=["Falls"]
    dict_falls=dictionary_creation(summary,train_data)
    #Dictionary for Fires and Explosion
    summary=["Fires and Explosion"]
    dict_fires=dictionary_creation(summary,train_data)
    #Dictionary for other
    summary=["Other","Others"]
    dict_others=dictionary_creation(summary,train_data)
    #Dictionary for Struck By Moving Objects
    summary=["Struck By Moving Objects"]
    dict_struck=dictionary_creation(summary,train_data)
    #Dictionary for Suffocation
    summary=["Suffocation"]
    dict_suffocation=dictionary_creation(summary,train_data)
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Classification for the Test File(Test_Masia.csv) using NEW MODEL
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    test_data=necessary.pd.read_csv('/Users/AJ/PycharmProjects/Assignment_Text_Mining/Test_Masia.csv')
    col_name_1=test_data.columns[0]
    test_data=test_data.rename(columns = {col_name_1:'Cause'})
    predictions=[]
    predictions=process_test_data(test_data,dict_caught_btw_objects,dict_collapse_of_object,dict_drowning,dict_electrocution,dict_chemical_substances,dict_extereme_temperature,dict_falls,dict_fires,dict_others,dict_struck,dict_suffocation)
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Prediction and accuracy report  for the NEW MODEL
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    actual_classifications=list(test_data.Cause)
    actuals = necessary.pd.Series(actual_classifications)
    predicted = necessary.pd.Series(predictions)
    confusion_matrix=necessary.pd.crosstab(actuals, predicted, rownames=['Actuals'], colnames=['Predicted'], margins=True)
    target_names = ["Caught in/between Objects","Collapse of object","Drowning","Electrocution","Exposure to Chemical Substances","Exposure to extreme temperatures","Falls","Fires and Explosion","Other","Struck By Moving Objects","Suffocation"]
    print(classification_report(actuals, predicted, target_names=target_names))
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Classification for the Test File(osha.csv) using NEW MODEL
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    test_data = necessary.pd.read_csv('/Users/AJ/PycharmProjects/Assignment_Text_Mining/osha.csv')
    col_name_1=test_data.columns[0]
    test_data=test_data.rename(columns = {col_name_1:'Cause'})
    predictions=[]
    predictions=process_test_data(test_data,dict_caught_btw_objects,dict_collapse_of_object,dict_drowning,dict_electrocution,dict_chemical_substances,dict_extereme_temperature,dict_falls,dict_fires,dict_others,dict_struck,dict_suffocation)
    predictions_str='", "'.join(predictions)
    prediction_new=[[x] for x in predictions]
    with open("output.csv", "wb") as f:
        writer = csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerow(["Classification","Occupation ","Activities prior accidents"])
        writer.writerows(prediction_new)
    print("The classification for test file(osha.csv) is in the file output.csv")
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Risky occupations
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    occupation=prior_occupation(test_data)
    occupation_single=[];
    for y in occupation:
        for x in y:
            occupation_single.append([x])
    with open("Occupations_new.csv","wb") as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerows(occupation_single)
    print("The results for the risky occupations is in the file Occupations_new.csv")
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Risky activities
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    activities_list=activities(test_data)
    activities_list=[[x] for x in activities_list]
    with open("Activities.csv","wb") as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerows(activities_list)
    print("The results for the risky activities is given in thr Activities.csv file")
    '''
    -------------------------------------------------------------------------------------------------------------------------------------
    Risky tools
    -------------------------------------------------------------------------------------------------------------------------------------
    '''
    tools_list=tools(test_data)
    tools_list_single=[[x] for x in tools_list]
    with open("Tools.csv","wb") as f:
        writer=csv.writer(f,quoting=csv.QUOTE_ALL)
        writer.writerows(tools_list_single)
    print("The results for the risky objects that causes accident is given in thr Tools.csv file")


