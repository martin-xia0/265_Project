import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class HyperClassifier:
    
    def load_data(self):
        # load dataset
        df = pd.read_csv("pine_data.csv")

        n_pixels = 200
        pixel_tag = ['px'+str(i+1) for i in range(n_pixels)]

        features = pixel_tag
        # iamge pixels
        self.x = df.loc[:, features].values
        # image class
        self.y = df.loc[:,['target']].values

        # standardizing the features
        from sklearn.preprocessing import MinMaxScaler
        scaler_model = MinMaxScaler()
        scaler_model.fit(self.x.astype(float))
        self.x = scaler_model.transform(self.x)

    def pca_analysis(self, n_dim):
        from sklearn.decomposition import PCA

        # finding the principle components
        pca = PCA(n_components=n_dim)
        principal_components = pca.fit_transform(self.x)
        ev = pca.explained_variance_ratio_
        # bar graph for explained variance ratio
        plt.bar([i for i in range(n_dim)], list(ev*100), label='Principal Components', color='b')
        plt.legend()
        plt.xlabel('Principal Components')
        x_label = ['PC'+str(i+1) for i in range(10)]
        plt.xticks([1,2,3,4,5,6,7,8,9,10], x_label, fontsize=8, rotation=30)
        plt.ylabel('Variance Ratio')
        plt.title('Variance Ratio of INDIAN PINES Dataset')
        plt.show()

    def dimensional_reduction(self, n_dim=20):
        """
        PCA reduction
        """
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_dim)
        principal_components = pca.fit_transform(self.x)
        principal_df = pd.DataFrame(data=principal_components)
        target_df = pd.DataFrame(data=self.y, columns=['target'])
        self.processed_df = pd.concat([principal_df, target_df], axis=1)


    def direct_knn(self):
        print(">>> Direct knn classification <<<")
        self.knn_classification(self.x, self.y)
    
    def knn_after_pca_general(self):
        acc_list = []
        t_list = []
        dim_set = [i for i in range(5, 100, 1)]
        for n_dim in dim_set:
            t, accuracy =  hyp.knn_after_pca(n_dim)
            acc_list.append(accuracy)
            t_list.append(t)
        plt.xlabel('N dimensions')
        plt.ylabel('Accuracy')
        plt.title('Dimensions VS accuracy')
        plt.plot(dim_set, acc_list)
        plt.show()
        
        plt.xlabel('N dimensions')
        plt.ylabel('Time')
        plt.title('Dimensions VS time')
        plt.plot(dim_set, t_list)
        plt.show()

    def knn_after_pca(self, n_dim):
        print(">>> KNN classification after dimensional reduction to {} <<<".format(n_dim))
        self.dimensional_reduction(n_dim)
        X = self.processed_df.drop(['target'], axis=1)
        y = self.processed_df['target']
        return self.knn_classification(X, y)

        
    def knn_classification(self, X, y):
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn import metrics
        import time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        
        model=KNeighborsClassifier(n_neighbors =13, weights='uniform', algorithm='auto')
        model.fit(X_train, y_train)
        start = time.time()
        Yhat = model.predict(X_test)
        end = time.time()
        t = end - start
        accuracy = metrics.accuracy_score(Yhat, y_test)*100
        print('Time Taken For Classification is :', t)
        print("Accuracy :", accuracy)
        return t, accuracy



if __name__=="__main__":
    hyp = HyperClassifier()
    hyp.load_data()
    hyp.knn_after_pca_general()
    hyp.direct_knn()