import pickle
def personality(l1):
    loaded_model = pickle.load(open('C:\\Users\\91996\\Personality-prediction-system\\finalized_model.sav', 'rb'))
    #Xnew = [[0,24,0.75,5,6,7,2]]
    y_pred = loaded_model.predict(l1)
    print(y_pred)
    return y_pred