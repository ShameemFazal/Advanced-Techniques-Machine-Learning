class loadingPreprocessingFunctions:
    
        def loadingandPreprocessing(preprocessingBoth,RandD_Spend,Administration,Marketing_Spend,State_Florida,State_New_York):
            import pickle
            import pandas as pd
            dataset = pd.read_csv("50_Startups.csv")
            dataset = pd.get_dummies(dataset,drop_first=True)
            Independent = dataset[["R&D Spend","Administration","Marketing Spend","State_Florida","State_New York"]]
            dependent = dataset[["Profit"]]
            from sklearn.model_selection import train_test_split
            X_train,X_test,Y_train,Y_test =train_test_split(Independent,dependent,test_size=0.30, random_state=0)
            from sklearn.preprocessing import StandardScaler
            if(preprocessingBoth=='Yes'):
                sc=StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
                scy = StandardScaler()
                Y_train = scy.fit_transform(Y_train)
                Y_test = scy.transform(Y_test)
            else:
                sc=StandardScaler()
                X_train = sc.fit_transform(X_train)
                X_test = sc.transform(X_test)
            preprocessed_input = sc.transform([[RandD_Spend,Administration,Marketing_Spend,State_Florida,State_New_York]])
            loaded_model = pickle.load(open("finalized_model_svr.sav",'rb'))
            output = loaded_model.predict(preprocessed_input)
            if(preprocessingBoth=='Yes'):
                output = scy.inverse_transform([output])
            else:
                output
            return output
 
 